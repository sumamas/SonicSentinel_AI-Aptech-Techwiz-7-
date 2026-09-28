"""Experimental full-recording timeline; the v5 clip benchmark evaluates one energy-selected 3-second window."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.audio.preprocess import load_audio_native,preprocess_signal
from src.audio.features import extract_features_from_signal,log_mel_from_signal
from src.audio.quality import assess_quality
from src.ml.predict import get_selected_model_name,score_arrays,model_digest,MODEL_KEYS
from src.ml.reliability import format_probabilities,decision_for,read_policy
from src.ml.training_config import DURATION_SECONDS

def spans(length,sr):
    # Non-overlapping windows with an explicit short-tail entry; timestamps are
    # source window boundaries, not inferred event onset/offset annotations.
    size=int(round(sr*DURATION_SECONDS))
    return [(start,min(start+size,length)) for start in range(0,length,size)]

def timeline(path,model='selected'):
    key=get_selected_model_name() if model=='selected' else model
    native,sr=load_audio_native(str(path));duration=len(native)/sr
    if not .3<=duration<=60 or not np.isfinite(native).all():raise ValueError('Provide finite audio between 0.30 and 60 seconds.')
    rows=[];signals=[];scored=[];policy=read_policy(ROOT)
    for start,end in spans(len(native),sr):
        raw=native[start:end];q=assess_quality(raw)
        row={'start_seconds':start/sr,'end_seconds':end/sr,'quality':q}
        if len(raw)/sr<.3 or q['label']=='Unusable':
            row.update(status='Not scored',reason='short_tail' if len(raw)/sr<.3 else 'unusable_audio',manual_review_required=True)
        else:
            signal,model_sr=preprocess_signal(raw,sr);signals.append(signal);scored.append(len(rows))
            row['status']='Scored'
        rows.append(row)
    if signals:
        X=np.vstack([extract_features_from_signal(y,model_sr) for y in signals]) if key!='custom_cnn' else None
        mels=np.asarray([log_mel_from_signal(y,model_sr)[...,None] for y in signals]) if key=='custom_cnn' else None
        labels,probs=score_arrays(key,X,mels)
        for index,p in zip(scored,probs):
            result=format_probabilities(labels,p)
            result['decision']=decision_for(result,rows[index]['quality']['label'],policy,['timeline_not_independently_validated'])
            rows[index].update(result)
    return {'model':key,'model_sha256':model_digest(key),'duration_seconds':duration,
            'analysis':'Experimental consecutive 3-second windows; source timestamps, no event-boundary detection.',
            'validated_end_to_end':False,'gtm_comparison':'Not performed','active_alert':False,
            'manual_review_required':True,'windows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audio');p.add_argument('--model',choices=['selected',*MODEL_KEYS],default='selected');p.add_argument('--output');a=p.parse_args()
    result=timeline(a.audio,a.model);text=json.dumps(result,indent=2);print(text)
    if a.output:
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding='utf-8')
