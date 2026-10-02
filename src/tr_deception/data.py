"""Strict parameter, sample and participant-split schemas for prepared inputs."""
import csv
import math
import numbers
from pathlib import Path
import re
import numpy as np
import yaml
from tr_deception.common import BASE_C, RHO, BOUND, SEED, BOOTSTRAP_DRAWS

class ConfigError(ValueError):
    """Invalid paper configuration."""

class _Loader(yaml.SafeLoader):
    """YAML loader that rejects duplicate fields."""

def _mapping(loader,node,deep=False):
    output={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node,deep=deep)
        if not isinstance(key,str) or key in output:
            raise ConfigError('Configuration keys must be unique strings')
        output[key]=loader.construct_object(value_node,deep=deep)
    return output

_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,_mapping)

def load_config(path):
    with Path(path).open(encoding='utf-8') as stream:cfg=yaml.load(stream,Loader=_Loader)
    if not isinstance(cfg,dict) or cfg.get('dataset') not in ('DOLOS','MU3D'):
        raise ConfigError('dataset must be DOLOS or MU3D')
    expected=dict(embedding_dim=768,K=3 if cfg['dataset']=='DOLOS' else 1,base_C=BASE_C,
        rho=RHO,bootstrap_draws=BOOTSTRAP_DRAWS,seed=SEED)
    if cfg['dataset']=='MU3D':expected['outer_folds']=5
    allowed=set(expected)|{'dataset','delta'}
    if set(cfg)!=allowed:
        raise ConfigError('Configuration fields differ; missing='+str(sorted(allowed-set(cfg)))+'; unknown='+str(sorted(set(cfg)-allowed)))
    for key,value in expected.items():
        actual=cfg[key]
        if isinstance(value,int):
            if type(actual) is not int:raise ConfigError(key+' must be an integer')
        elif isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(actual):
            raise ConfigError(key+' must be finite numeric')
        if actual!=value:raise ConfigError(key+' differs from the paper setting')
    delta=cfg['delta']
    if delta!='ln2' and (isinstance(delta,bool) or not isinstance(delta,(int,float)) or not math.isfinite(delta) or not math.isclose(delta,BOUND,rel_tol=0,abs_tol=1e-15)):
        raise ConfigError('delta must equal ln2')
    cfg['delta']=BOUND
    return cfg

def validate_id(value,field):
    if isinstance(value,str) and value.strip():return value
    if isinstance(value,numbers.Integral) and not isinstance(value,(bool,np.bool_)):return str(value)
    raise ValueError(field+' must be a nonempty string or integer ID')

def validate_label(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral) or value not in (0,1):
        raise ValueError('label must be integer 0=truthful or 1=deceptive')
    return int(value)

def validate_training_inputs(X,labels,participant_ids):
    x=np.asarray(X,dtype=np.float64)
    y=[validate_label(v) for v in labels]
    ids=[validate_id(v,'participant_id') for v in participant_ids]
    if x.ndim!=2 or x.shape[1]<1 or not np.isfinite(x).all() or len(x)!=len(y) or len(y)!=len(ids):
        raise ValueError('training matrix, labels and participant IDs must have matching finite rows')
    if len(set(ids))<2 or set(y)!={0,1}:
        raise ValueError('training requires at least two participants and both classes')

def csv_records(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames)!=len(set(reader.fieldnames)):
            raise ValueError('CSV requires unique field names')
        return list(reader)

def _integer_token(value,field):
    if not isinstance(value,str) or re.fullmatch(r'0|[1-9][0-9]*',value) is None:
        raise ValueError(field+' must use a nonnegative decimal integer token')
    return int(value)

def _boolean(value):
    if isinstance(value,str) and value.lower() in ('true','false','0','1'):
        return value.lower() in ('true','1')
    raise ValueError('boolean fields must be true/false or 1/0')

def load_dataset(sample_file,embedding_file):
    """Load a metadata CSV and floating-point NPY [N,768] by explicit embedding_row."""
    matrix=np.load(embedding_file,allow_pickle=False)
    if matrix.ndim!=2 or matrix.shape[1]!=768 or not np.issubdtype(matrix.dtype,np.floating) or not np.isfinite(matrix).all():
        raise ValueError('embeddings must be finite floating-point [N,768]')
    data={};used=set()
    required={'sample_id','participant_id','label','embedding_row','truthful_reference_eligibility'}
    for row in csv_records(sample_file):
        if not required<=set(row):raise ValueError('sample CSV is missing required fields')
        sid=validate_id(row['sample_id'],'sample_id');pid=validate_id(row['participant_id'],'participant_id')
        if sid in data:raise ValueError('duplicate sample_id')
        label=validate_label(_integer_token(row['label'],'label'))
        index=_integer_token(row['embedding_row'],'embedding_row')
        if index>=len(matrix) or index in used:raise ValueError('embedding_row must be in range and unique')
        used.add(index);x=np.asarray(matrix[index],dtype=np.float64)
        if abs(np.linalg.norm(x)-1)>1e-3:raise ValueError('embeddings must already be L2 normalized')
        eligible=_boolean(row['truthful_reference_eligibility'])
        if eligible and label!=0:raise ValueError('eligible references must be known truthful')
        data[sid]=dict(sample_id=sid,participant_id=pid,y=label,x=x,reference_eligible=eligible)
    if not data:raise ValueError('sample CSV is empty')
    return data

def participant_splits(cfg,data,path):
    """Read supplied participant roles; reject overlap and incorrect paper counts."""
    records=csv_records(path);known={r['participant_id'] for r in data.values()}
    for r in records:r['participant_id']=validate_id(r.get('participant_id'),'participant_id')
    if cfg['dataset']=='DOLOS':
        if len({r['participant_id'] for r in records})!=len(records):raise ValueError('duplicate participant role row')
        train={r['participant_id'] for r in records if _boolean(r['base_train'])}
        residual={r['participant_id'] for r in records if _boolean(r['residual_train'])}
        test={r['participant_id'] for r in records if _boolean(r['evaluation'])}
        if (len(train),len(residual),len(test))!=(30,7,16):raise ValueError('DOLOS participant counts must be 30/7/16')
        counts=tuple(sum(r['participant_id'] in ids for r in data.values()) for ids in [train,residual,test])
        if counts!=(197,50,425):raise ValueError('DOLOS response counts must be 197/50/425')
        splits=[('dolos',train,residual,test)]
    else:
        if len(known)!=80 or len(data)!=320:raise ValueError('MU3D requires 80 participants and 320 responses')
        for p in known:
            labels=[r['y'] for r in data.values() if r['participant_id']==p]
            if labels.count(0)!=2 or labels.count(1)!=2:raise ValueError('MU3D requires two truthful and two deceptive responses per participant')
        for r in records:
            r['outer_fold']=_integer_token(r['outer_fold'],'outer_fold');r['role']=r['role'].lower()
            if r['outer_fold'] not in range(5) or r['role'] not in ('train','test'):raise ValueError('invalid outer fold or role')
        splits=[]
        for fold in range(5):
            rows=[r for r in records if r['outer_fold']==fold]
            if len({r['participant_id'] for r in rows})!=len(rows):raise ValueError('duplicate participant within outer fold')
            train={r['participant_id'] for r in rows if r['role']=='train'}
            test={r['participant_id'] for r in rows if r['role']=='test'}
            if len(train)!=64 or len(test)!=16 or train|test!=known:raise ValueError('MU3D fold must partition 64 training and 16 test participants')
            splits.append((str(fold),train,train,test))
        evaluated=[p for _,_,_,test in splits for p in test]
        if len(evaluated)!=80 or len(set(evaluated))!=80:raise ValueError('each participant must be tested once')
    for _,train,residual,test in splits:
        if train&test or not residual<=train or not (train|test)<=known:raise ValueError('invalid participant fitting/evaluation scopes')
    return splits
