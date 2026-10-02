"""Participant-independent fitting and aggregate evaluation from prepared inputs."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from tr_deception.common import SEED, RESIDUAL_FOLDS, BOOTSTRAP_DRAWS, require_disjoint, participant_folds
from tr_deception.base import fit_base_model
from tr_deception.residual import fit_residual, fit_shared_residual
from tr_deception.evaluation import draw_metrics, participant_metric, participant_macro_metrics, paired_participant_bootstrap
from tr_deception.fusion import fit_score_fusion
from tr_deception.data import load_config, load_dataset, participant_splits, csv_records

def rows_for(data, participants):
    return sorted((r for r in data.values() if r['participant_id'] in set(participants)), key=lambda r: r['sample_id'])

def rows_for_participant(data, participant):
    return rows_for(data, {participant})

def fit_head(data, participants, head='LR'):
    rows = rows_for(data, participants)
    if not rows:
        raise ValueError('empty fitting scope')
    from tr_deception.data import validate_training_inputs
    validate_training_inputs(np.stack([r['x'] for r in rows]), [r['y'] for r in rows], [r['participant_id'] for r in rows])
    if head == 'LR':
        return fit_base_model(np.stack([r['x'] for r in rows]), [r['y'] for r in rows], [r['participant_id'] for r in rows])
    if head == 'SVM':
        from tr_deception.heads import fit_calibrated_svm
        return fit_calibrated_svm(rows, participants)
    if head == 'MLP':
        from tr_deception.heads import fit_mlp
        if any((r['x'].size != 768 for r in rows)):
            raise ValueError('frozen MLP architecture requires768-dimensional embeddings')
        return fit_mlp(rows, participants)
    raise ValueError('head must be LR/SVM/MLP')

def oof_base_logits(data, participants, head='LR', folds=RESIDUAL_FOLDS):
    logits = {}
    for _, train, held in participant_folds(participants, folds):
        require_disjoint(train, held)
        model = fit_head(data, train, head)
        rows = rows_for(data, held)
        z = model.logits(np.stack([r['x'] for r in rows]))
        logits.update({r['sample_id']: np.asarray(v, dtype=np.float64) for r, v in zip(rows, z)})
    if set(logits) != {r['sample_id'] for r in rows_for(data, participants)}:
        raise ValueError('OOF scores do not cover exactly the training scope')
    return logits

def reference_draws(rows, k):
    refs = [r for r in rows if r['reference_eligible']]
    if len(refs) < k:
        raise ValueError('insufficient known truthful references')
    for index, combo in enumerate(itertools.combinations(refs, k)):
        chosen = {r['sample_id'] for r in combo}
        targets = [r for r in rows if r['sample_id'] not in chosen]
        if not targets:
            raise ValueError('reference draw has no query responses')
        yield (index, combo, targets)

def draw_q(rows, reference_rows):
    from tr_deception.self_reference import build_batch_self_reference_features
    if not rows or not reference_rows:
        raise ValueError('queries and truthful references must be nonempty')
    return build_batch_self_reference_features(np.stack([r['x'] for r in rows]), np.stack([r['x'] for r in reference_rows]))

def fit_reference_residual(data, participants, oof, k, head='LR'):
    q, z, y, pids = ([], [], [], [])
    for p in sorted(participants):
        for _, refs, targets in reference_draws(rows_for_participant(data, p), k):
            q.extend(draw_q(targets, refs))
            z.extend((oof[r['sample_id']] for r in targets))
            y.extend((r['y'] for r in targets))
            pids.extend([p] * len(targets))
    if not q:
        raise ValueError('empty residual fitting data')
    if head == 'MLP':
        return fit_shared_residual(q, z, y, pids)
    return fit_residual(q, z, y, pids)

def probability_scores(z0, correction=None):
    z = np.asarray(z0, dtype=np.float64)
    if correction is not None:
        z = z + (np.asarray(correction)[:, None] if z.ndim == 2 else np.asarray(correction))
    p = expit(z)
    return p.mean(axis=1, dtype=np.float64) if z.ndim == 2 else p

def evaluate_reference_model(data, participants, model, residual, k):
    metrics = {}
    draws = 0
    queries = 0
    for p in sorted(participants):
        participant_draws = []
        for _, refs, targets in reference_draws(rows_for_participant(data, p), k):
            x = np.stack([r['x'] for r in targets])
            q = draw_q(targets, refs)
            z0 = model.logits(x)
            correction = residual.correction(q)
            participant_draws.append(draw_metrics([r['y'] for r in targets], probability_scores(z0), probability_scores(z0, correction)))
            draws += 1
            queries += len(targets)
        metrics[p] = participant_metric(participant_draws)
    return (metrics, draws, queries)

def summarize(participants, draws=BOOTSTRAP_DRAWS):
    result = participant_macro_metrics(participants)
    bs = paired_participant_bootstrap(participants, draws=draws, seed=SEED)
    result.update(delta_ap_ci=bs['ap']['ci'], delta_auroc_ci=bs['auroc']['ci'], participants=len(participants), bootstrap_draws=draws, seed=SEED)
    return result

def run_participant_experiment(data, train, residual_participants, test, k, head='LR', bootstrap_draws=BOOTSTRAP_DRAWS):
    train, residual_participants, test = (set(train), set(residual_participants), set(test))
    require_disjoint(train, test)
    if not residual_participants <= train:
        raise ValueError('residual fitting participants must be a training subset')
    oof = oof_base_logits(data, train, head)
    residual = fit_reference_residual(data, residual_participants, oof, k, head)
    final = fit_head(data, train, head)
    parts, draws, queries = evaluate_reference_model(data, test, final, residual, k)
    result = summarize(parts, bootstrap_draws)
    result.update(head=head, K=k, reference_draws=draws, query_rows=queries, train_participants=len(train), residual_participants=len(residual_participants))
    return (result, parts)

def read_scores(path):
    values = {}
    for r in csv_records(path):
        key = (str(r['context']), str(r['sample_id']))
        v = float(r['probability'])
        if key in values or not np.isfinite(v) or (not 0 <= v <= 1):
            raise ValueError('duplicate/invalid external score')
        values[key] = v
    return values

def _scores(values, context, rows):
    try:
        return np.array([values[context, r['sample_id']] for r in rows], dtype=np.float64)
    except KeyError as error:
        raise ValueError('external score context does not cover required samples') from error

def crossfit_score_fusion(data, train, residual_participants, test, k, contexts, semantic, visual):
    train, residual_participants, test = (set(train), set(residual_participants), set(test))
    require_disjoint(train, test)
    if not residual_participants <= train:
        raise ValueError('residual scope is not a training subset')
    oof = {}
    seen = set()
    for spec in contexts['inner']:
        held = set(spec['held_participants'])
        fit = train - held
        context = spec['context']
        if not held <= train or seen & held or (not fit):
            raise ValueError('inner held scopes must partition training participants')
        if set(spec['fit_participants']) != fit:
            raise ValueError('score-generator fitting scopes must equal the corresponding inner training scope')
        seen |= held
        require_disjoint(fit, held)
        fitrows = rows_for(data, fit)
        heldrows = rows_for(data, held)
        ps = _scores(semantic, context, fitrows)
        pf = _scores(visual, context, fitrows)
        fusion = fit_score_fusion(ps, pf, [r['y'] for r in fitrows], [r['participant_id'] for r in fitrows])
        z = fusion.logits(_scores(semantic, context, heldrows), _scores(visual, context, heldrows))
        oof.update({r['sample_id']: float(v) for r, v in zip(heldrows, z)})
    if seen != train:
        raise ValueError('inner held scopes do not cover all training participants')
    residual = fit_reference_residual(data, residual_participants, oof, k)
    full = contexts['full']
    if set(full['fit_participants']) != train or set(full['evaluation_participants']) != test:
        raise ValueError('full score scope differs from outer participant split')
    context = full['context']
    fitrows = rows_for(data, train)
    fusion = fit_score_fusion(_scores(semantic, context, fitrows), _scores(visual, context, fitrows), [r['y'] for r in fitrows], [r['participant_id'] for r in fitrows])
    parts = {}
    for p in sorted(test):
        dm = []
        for _, refs, queries in reference_draws(rows_for_participant(data, p), k):
            z = fusion.logits(_scores(semantic, context, queries), _scores(visual, context, queries))
            q = draw_q(queries, refs)
            dm.append(draw_metrics([r['y'] for r in queries], expit(z), expit(residual.logits(z, q))))
        parts[p] = participant_metric(dm)
    return parts

def run_semantic(data, splits, config, head='LR'):
    participants={}
    for _,train,residual,test in splits:
        _,parts=run_participant_experiment(data,train,residual,test,config['K'],head,config['bootstrap_draws'])
        if participants.keys() & parts.keys():raise ValueError('evaluation participant repeated across folds')
        participants.update(parts)
    result=summarize(participants,config['bootstrap_draws'])
    result.update(dataset=config['dataset'],head=head,K=config['K'])
    return result

def run_fusion(data, splits, config, semantic_scores, visual_scores, contexts):
    participants={}
    for fold,train,residual,test in splits:
        context=contexts[fold]
        count=5 if config['dataset']=='DOLOS' else 4
        held_count=6 if config['dataset']=='DOLOS' else 16
        if len(context['inner'])!=count or any(len(set(c['held_participants']))!=held_count for c in context['inner']):
            raise ValueError('fusion contexts have incorrect inner fold or held-participant counts')
        parts=crossfit_score_fusion(data,train,residual,test,config['K'],context,semantic_scores,visual_scores)
        if participants.keys() & parts.keys():raise ValueError('evaluation participant repeated across folds')
        participants.update(parts)
    result=summarize(participants,config['bootstrap_draws'])
    result.update(dataset=config['dataset'],head='SF',K=config['K'])
    return result

def _path(value):
    return Path(value).expanduser().resolve()

def _parser(description):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--config',required=True,type=_path)
    parser.add_argument('--samples',required=True,type=_path,help='Metadata CSV with IDs, labels, embedding_row and truthful_reference_eligibility')
    parser.add_argument('--embeddings',required=True,type=_path,help='Prepared L2-normalized float NPY matrix [N,768]')
    parser.add_argument('--splits',required=True,type=_path,help='Participant role/fold CSV')
    parser.add_argument('--output',required=True,type=_path,help='Aggregate JSON output')
    parser.add_argument('--overwrite',action='store_true',help='Replace a distinct existing output')
    return parser

def _check_output(output,inputs,overwrite):
    output=output.resolve()
    for value in inputs:
        source=value.resolve()
        if output==source or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError('input and output refer to the same file')
    if output.exists() and not overwrite:raise FileExistsError('output exists; use --overwrite explicitly')
    if output.exists() and not output.is_file():raise ValueError('output must be a file')

def _execute(parser,argv,mode):
    args=parser.parse_args(argv)
    inputs=[args.config,args.samples,args.embeddings,args.splits]
    if mode=='fusion':inputs += [args.semantic_scores,args.visual_scores,args.contexts]
    try:
        _check_output(args.output,inputs,args.overwrite)
        cfg=load_config(args.config)
        data=load_dataset(args.samples,args.embeddings)
        splits=participant_splits(cfg,data,args.splits)
        if mode=='fusion':
            contexts=json.loads(args.contexts.read_text(encoding='utf-8'))
            result=run_fusion(data,splits,cfg,read_scores(args.semantic_scores),read_scores(args.visual_scores),contexts)
        else:
            head=args.head.upper() if mode=='heads' else 'LR'
            result=run_semantic(data,splits,cfg,head)
        output=dict(dataset=result['dataset'],head=result['head'],participants=result['participants'],
            base_ap=result['base_ap'],tr_ap=result['self_ap'],delta_ap=result['delta_ap'],delta_ap_ci=result['delta_ap_ci'],
            base_auroc=result['base_auroc'],tr_auroc=result['self_auroc'],delta_auroc=result['delta_auroc'],
            delta_auroc_ci=result['delta_auroc_ci'],bootstrap_draws=cfg['bootstrap_draws'],seed=cfg['seed'])
        _check_output(args.output,inputs,args.overwrite)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('w' if args.overwrite else 'x',encoding='utf-8',newline='\n') as stream:
            json.dump(output,stream,indent=2);stream.write('\n')
        print(json.dumps(output,indent=2))
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
    return 0

def primary_main(argv=None):
    return _execute(_parser('Run participant-independent LR and LR + TR'),argv,'primary')

def heads_main(argv=None):
    parser=_parser('Run SVM/MLP with shared truthful-reference correction')
    parser.add_argument('--head',choices=['svm','mlp'],required=True)
    return _execute(parser,argv,'heads')

def fusion_main(argv=None):
    parser=_parser('Run SF and SF + TR from externally supplied context-specific probabilities')
    parser.add_argument('--semantic-scores',type=_path,required=True)
    parser.add_argument('--visual-scores',type=_path,required=True)
    parser.add_argument('--contexts',type=_path,required=True,help='JSON fitting/held/evaluation contexts indexed by outer fold')
    return _execute(parser,argv,'fusion')
