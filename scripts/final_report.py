"""Report dataset and held-out metrics without equating them to full SRS acceptance."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.ml.predict import MODEL_KEYS,get_selected_model_name,model_digest
from src.ml.training_config import REPORTS_DIR

def main():
    dataset=json.loads((ROOT/'COPY_COMPLETE.json').read_text())
    selected=get_selected_model_name();models={}
    for key in MODEL_KEYS:
        p=REPORTS_DIR/f'{key}_test_metrics.json';m=json.loads(p.read_text())
        if m['model_sha256']!=model_digest(key):raise ValueError('Stale test evidence: '+key)
        checks=m['srs_metric_status'];models[key]={
            'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
            'critical_recall_per_class':checks['critical_recall_per_class'],
            'raw_metric_targets_pass':all(checks[k] for k in ['all_ten_classes','accuracy_at_least_0_85','macro_f1_at_least_0_80','every_critical_class_recall_at_least_0_85']),
            'policy_assessment':m['policy_assessment'],
            'model_sha256':m['model_sha256']}
    result={'selected_using_validation':selected,'dataset_files':dataset['retained_unique_files'],
            'dataset_all_ten_300_files':dataset['all_ten_300_file_target'],
            'recording_source_independence_verified':dataset['source_independence_verified'],
            'models':models,'full_srs_verified':False,
            'remaining_checks':['Audit unreviewed semantic labels, especially Machinery Fault vs normal machinery, Aggression vs scream/help, and mixed-event recordings.',
                'Verify recording-source/speaker/session groups; hashes do not identify all crops or transcodes.',
                'Historical test recordings have been evaluated before. New-to-metadata recordings are not proof of untouched independent test sources.',
                'V5 clip benchmark uses the highest-energy 3-second window after trimming. Energy is not an event label; the full-audio timeline needs event-level validation.',
                'GTM must be trained independently on training audio, then compared on at least 100 unseen examples (at least 10/class).',
                'Noise tests are synthetic post-preprocessing white noise; assess field microphone/noise, false alarms and per-critical-class gated recall.',
                'Validate app live latency, database/auth/dashboard and alert confirmation rules before claiming complete SRS compliance.']}
    (ROOT/'reports/final_readiness.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Model                  Accuracy    Macro F1    Raw metric targets')
    for key,m in models.items():print(f"{key:23}{m['accuracy']:.4f}      {m['macro_f1']:.4f}      {m['raw_metric_targets_pass']}")
    print('\nSelected:',selected,'(validation only)')
    for label,value in models[selected]['critical_recall_per_class'].items():print(f'{label}: {value:.4f} (target >= 0.85)')
    print('300 unique files/class:',result['dataset_all_ten_300_files'])
    print('Full SRS verified: NO. See reports/final_readiness.json for remaining evidence.')
    return result
if __name__=='__main__':main()
