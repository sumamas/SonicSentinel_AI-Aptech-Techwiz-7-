"""Run the ordered, versioned training workflow and back up each completed stage."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.ml.provenance import package_version
from src.ml.reliability import sha256_file

COPY_DIRS=['src','scripts','config','models','reports','data/splits','data/features','source_project_review']
COPY_FILES=['data/metadata.csv','COPY_COMPLETE.json','requirements-colab.txt','requirements-windows-ml.txt','requirements-ml.txt','README_TRAINED.md','SRS_STATUS.md','START_HERE.md','KIT_VERIFICATION.md','release_manifest.json']

def atomic_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.copying')
    shutil.copyfile(src,tmp);tmp.replace(dst)

def backup_to(destination):
    if not destination:return
    destination=Path(destination)
    if destination==ROOT or destination.is_relative_to(ROOT):raise ValueError('Backup must be outside the project.')
    for relative in COPY_DIRS:
        for path in (ROOT/relative).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':atomic_copy(path,destination/path.relative_to(ROOT))
    for relative in COPY_FILES:
        if (ROOT/relative).is_file():atomic_copy(ROOT/relative,destination/relative)
    print('BACKUP:',destination,flush=True)

def save_versions():
    packages=['numpy','scipy','scikit-learn','librosa','joblib','pandas','soundfile','tensorflow','keras','numba','llvmlite','matplotlib','imageio-ffmpeg']
    versions={p:package_version(p) for p in packages};report=ROOT/'reports';report.mkdir(exist_ok=True)
    (report/'training_versions.json').write_text(json.dumps({'python':sys.version,'packages':versions},indent=2))
    text='\n'.join(k+'=='+v for k,v in versions.items())+'\n'
    (report/'requirements-actual-ml.txt').write_text(text)
    # Export the actual pinned inference environment for Windows, without CUDA.
    (ROOT/'requirements-windows-ml.txt').write_text(text)
    print('Training versions:',versions,flush=True)
    return versions

def export_zip(destination):
    out=Path(destination or ROOT.parent)/('SonicSentinel_10Class_Trained_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for relative in ['src','scripts','config','models','reports','data/splits','source_project_review']:
            for p in sorted((ROOT/relative).rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':z.write(p,p.relative_to(ROOT).as_posix())
        for relative in COPY_FILES:
            p=ROOT/relative
            if p.is_file():z.write(p,relative)
    print('TRAINED ZIP:',out,flush=True)
    review=out.with_name(out.name.replace('_Trained_','_Results_'))
    with zipfile.ZipFile(out) as original,zipfile.ZipFile(review,'x',zipfile.ZIP_DEFLATED) as z:
        for entry in original.infolist():
            if not entry.filename.startswith('models/') or entry.filename.endswith('.json'):
                z.writestr(entry,original.read(entry.filename))
    print('RESULTS ZIP (small, no trained weights):',review,flush=True)
    return out

def run(command,log):
    print('\nRUNNING:', ' '.join(command),flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env.setdefault('TF_CPP_MIN_LOG_LEVEL','2');env.setdefault('OMP_NUM_THREADS','2');env.setdefault('TF_NUM_INTRAOP_THREADS','2');env.setdefault('TF_NUM_INTEROP_THREADS','2');env['MPLBACKEND']='Agg'
    with log.open('a',encoding='utf-8') as stream,subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
        for line in process.stdout:print(line,end='',flush=True);stream.write(line);stream.flush()
        rc=process.wait()
    if rc:raise RuntimeError(f'Stage failed ({rc}). See {log}. The next stage was not started.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--quick',action='store_true');p.add_argument('--backup');p.add_argument('--skip-noise',action='store_true');a=p.parse_args()
    if a.epochs<1 or a.batch_size<1:raise ValueError('Epochs and batch size must be positive.')
    if a.backup:
        destination=Path(a.backup).resolve();destination.mkdir(parents=True,exist_ok=True);os.environ['SONIC_BACKUP_DIR']=str(destination)
    else:destination=None
    run(['scripts/verify_prepared.py'],ROOT/'reports/logs/verification.log')
    versions=save_versions();q=['--quick'] if a.quick else [];noise=[] if a.skip_noise else ['--noise']
    code=[p for parent in ['src','scripts','config'] for p in (ROOT/parent).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    signature=hashlib.sha256(json.dumps({'metadata':sha256_file(ROOT/'data/metadata.csv'),'splits':{s:sha256_file(ROOT/f'data/splits/{s}.csv') for s in ['train','validation','test']},'code':{str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(code)},'versions':versions,'quick':a.quick,'epochs':a.epochs,'batch_size':a.batch_size,'noise':not a.skip_noise},sort_keys=True).encode()).hexdigest()
    state_path=ROOT/'reports/training_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'signature':signature,'stages':{}}
    if state['signature']!=signature:raise ValueError('This run has different inputs/code/settings. Prepare a new run folder instead of mixing evidence.')
    stages=[
      ('features',['scripts/build_feature_cache.py','--augment-train','1'],[f'data/features/{s}.{e}' for s in ['train','validation','test'] for e in ['npz','json']]),
      ('random_forest',['-m','src.ml.train_rf',*q],['models/random_forest.joblib','reports/model_evaluation/random_forest_training_summary.json']),
      ('svm',['-m','src.ml.train_svm',*q],['models/svm.joblib','reports/model_evaluation/svm_training_summary.json']),
      ('custom_cnn',['-m','src.ml.train_cnn','--epochs',str(a.epochs),'--batch-size',str(a.batch_size)],['models/custom_cnn.keras','models/custom_cnn_last.keras','models/custom_cnn_manifest.json','reports/model_evaluation/custom_cnn_training_summary.json']),
      ('validation',['scripts/evaluate_models.py','--split','validation',*noise],[f'reports/model_evaluation/{m}_validation_metrics.json' for m in ['random_forest','svm','custom_cnn']]+(['reports/model_evaluation/validation_noise_robustness_summary.csv'] if noise else [])),
      ('selection',['scripts/compare_models.py'],['models/selected_model.json','reports/model_evaluation/model_comparison.csv']),
      ('test',['scripts/evaluate_models.py','--split','test',*noise],[f'reports/model_evaluation/{m}_test_metrics.json' for m in ['random_forest','svm','custom_cnn']]+(['reports/model_evaluation/test_noise_robustness_summary.csv'] if noise else [])),
      ('final_report',['scripts/final_report.py'],['reports/final_readiness.json'])]
    try:
        for name,command,outputs in stages:
            prior=state['stages'].get(name)
            if prior:
                if any(not (ROOT/f).is_file() or sha256_file(ROOT/f)!=h for f,h in prior['outputs'].items()):raise ValueError('Completed output changed: '+name+'. Start a separate run.')
                print('ALREADY COMPLETE:',name,flush=True);continue
            run(command,ROOT/f'reports/logs/{name}.log')
            state['stages'][name]={'completed_at':datetime.now(timezone.utc).isoformat(),'outputs':{f:sha256_file(ROOT/f) for f in outputs}}
            temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
            backup_to(destination)
        export_zip(destination)
    finally:backup_to(destination)

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError) as exc:raise SystemExit(str(exc)) from exc
