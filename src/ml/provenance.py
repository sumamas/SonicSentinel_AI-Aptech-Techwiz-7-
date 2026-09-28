"""Fail clearly when trained weights, preprocessing, labels or dependencies differ."""
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
from pathlib import Path
from src.ml.training_config import ROOT, CLASSES

INPUT_FILES=('src/audio/preprocess.py','src/audio/decode.py','src/audio/features.py','src/ml/training_config.py')

def package_version(name):
    try:return version(name)
    except PackageNotFoundError:
        if name=='tensorflow':
            try:return version('tensorflow-cpu')
            except PackageNotFoundError:return None
        return None

def input_contract(cnn=False):
    packages=['numpy','scipy','librosa','soundfile','scikit-learn','joblib','imageio-ffmpeg']
    if cnn:packages+=['tensorflow','keras']
    return {'schema':'sonic-submission-v5','classes':CLASSES,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUT_FILES},
            'packages':{p:package_version(p) for p in packages}}

def assert_input_contract(saved,cnn=False):
    """Classes and preprocessing source must match exactly (they change the model
    input). Package patch versions only produce a logged warning, so a slightly
    different local install does not silently disable predictions."""
    if not isinstance(saved,dict):
        raise RuntimeError('Model export has no input contract. Re-export the model with this code.')
    current=input_contract(cnn)
    if saved.get('schema')!=current['schema'] or list(saved.get('classes',[]))!=list(current['classes']):
        raise RuntimeError('Model classes/schema do not match this code. Do not mix old models with new code.')
    if saved.get('source_sha256')!=current['source_sha256']:
        raise RuntimeError('Audio preprocessing source files changed after training (src/audio/*.py or training_config.py). Restore them or retrain.')
    diff={k:(v,current['packages'].get(k)) for k,v in (saved.get('packages') or {}).items() if current['packages'].get(k)!=v}
    if diff:
        import logging
        logging.getLogger('sonicsentinel').warning('Package versions differ from training environment: %s', diff)

def training_evidence(cnn=False):
    from src.ml.reliability import sha256_file
    return {'input_contract':input_contract(cnn),
            'metadata_sha256':sha256_file(ROOT/'data/metadata.csv'),
            'split_sha256':{s:sha256_file(ROOT/f'data/splits/{s}.csv') for s in ['train','validation','test']}}
