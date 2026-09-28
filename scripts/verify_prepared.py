"""Verify the immutable prepared dataset before any training or evaluation."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.ml.training_config import CLASSES
from src.ml.reliability import sha256_file

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def verify(root=ROOT):
    root=Path(root)
    if (root/'COPY_INCOMPLETE').exists():raise ValueError('Preparation was interrupted. Use a complete output folder.')
    info=json.loads((root/'COPY_COMPLETE.json').read_text())
    if info.get('version')!='sonic-submission-v5':raise ValueError('Not a v5 prepared dataset.')
    meta=root/'data/metadata.csv'
    if sha256_file(meta)!=info['metadata_sha256']:raise ValueError('Metadata changed after preparation.')
    metadata=read(meta);all_rows=[];sets={};labels={}
    for split in ['train','validation','test']:
        p=root/f'data/splits/{split}.csv'
        if sha256_file(p)!=info['prepared_split_csv_sha256'][split]:raise ValueError('Frozen split CSV changed: '+split)
        part=read(p)
        if {r['class_label'] for r in part}!=set(CLASSES):raise ValueError('All ten classes must appear in '+split)
        for r in part:
            p=(root/r['relative_path']).resolve()
            if not p.is_relative_to((root/'data/raw').resolve()) or not p.is_file():raise ValueError('Missing/invalid audio: '+r['relative_path'])
            if sha256_file(p)!=r['sha256']:raise ValueError('Audio changed: '+r['relative_path'])
            if r.get('historical_split') and r['historical_split']!=split:raise ValueError('Historical recording moved between splits.')
            if r['split']!=split or r['original_or_augmented']!='original':raise ValueError('Invalid split or augmented original.')
            g=r['original_audio_id']
            if g in labels and labels[g]!=r['class_label']:raise ValueError('Conflicting labels in a recording source group.')
            labels[g]=r['class_label']
        sets[split]={field:{r[field] for r in part if r.get(field)} for field in ['audio_id','relative_path','sha256','source_sha256','pcm_sha256','original_audio_id','split_group_id']}
        all_rows+=part
        print(split,':',len(part),'originals — bytes and split checked',flush=True)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for field in sets[a]:
            if sets[a][field]&sets[b][field]:raise ValueError(f'{a}/{b} overlap: {field}')
    if len(all_rows)!=len(metadata) or len({r['audio_id'] for r in all_rows})!=len(metadata):raise ValueError('Duplicate or missing rows across splits.')
    by_id={r['audio_id']:r for r in metadata}
    if any(by_id.get(r['audio_id'])!=r for r in all_rows):raise ValueError('Metadata and split rows differ.')
    print('VERIFIED:',len(metadata),'recordings | 10 classes')
    print('300 unique files/class:',info['all_ten_300_file_target'])
    print('Source independence verified:',info['source_independence_verified'])
    return info

if __name__=='__main__':
    try:verify()
    except (ValueError,KeyError,OSError) as exc:raise SystemExit(str(exc)) from exc
