"""Run directly with the project environment; uses an isolated temporary database."""
if __name__ == "__main__":
    import os,sys,tempfile,io,json,re
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));os.chdir(root)
    os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='1',DATABASE_URL='sqlite:///:memory:')
    from app import create_app
    from app_config import Config
    from extensions import db
    from models_db import User,AudioEvent,EventAnalysis,EventAction
    with tempfile.TemporaryDirectory() as td:
     class TestConfig(Config):
      TESTING=True;SQLALCHEMY_DATABASE_URI='sqlite:///:memory:';UPLOAD_FOLDER=td
     app=create_app(TestConfig)
     with app.app_context():
      db.create_all()
      for email,role in [('owner@example.com','normal_user'),('other@example.com','normal_user'),('admin@example.com','administrator')]:
       u=User(full_name='Test user',email=email,role=role);u.set_password('local-test-123');db.session.add(u)
      db.session.commit()
     def login(email):
      c=app.test_client();assert c.post('/auth/login',data={'email':email,'password':'local-test-123'}).status_code==302;return c
     owner=login('owner@example.com');other=login('other@example.com');admin=login('admin@example.com')
     for route in ['/workspace/overview','/audio/upload','/workspace/batch-upload','/workspace/event-history','/workspace/alerts','/workspace/manual-review','/workspace/profile','/workspace/settings','/workspace/model-comparison','/workspace/model-comparison?model=svm','/workspace/model-comparison?model=random_forest']:
      r=owner.get(route);assert r.status_code==200,(route,r.status_code)
     sample=(root/'cat_model_input.wav').read_bytes()
     def upload():
      r=owner.post('/api/audio/inspect',data={'audio':(io.BytesIO(sample),'test.wav')});assert r.status_code==200,r.get_json();return r.get_json()
     result=upload();event_id=result['event_id'];prefix=f'/workspace/events/{event_id}';assert result['prediction_enabled'];assert len(result['all_confidences'])==10
     assert upload()['duplicate_of']==event_id
     for suffix in ['','/audio','/visuals','/analysis.json']:
      assert owner.get(prefix+suffix).status_code==200,suffix
      assert other.get(prefix+suffix).status_code==404,suffix
     visual=owner.get(prefix+'/visuals').get_json();assert visual['waveform'] and len(visual['spectrogram'])==100
     raw=owner.get(prefix+'/analysis.json').get_json();assert raw['analysis']['metadata']['sample_rate']==22050;assert raw['analysis']['prediction']['model']=='custom_cnn'
     assert owner.get('/workspace/model-evidence/custom_cnn/confusion.png').status_code==200
     def token(c):
      c.get('/workspace/profile')
      with c.session_transaction() as s:return s['csrf_token']
     ot=token(owner);at=token(admin)
     payload={'csrf_token':ot,'action':'correct','corrected_class':'Animal Sound','comment':'QA review','recommended_action':'Check source.'}
     assert owner.post(prefix+'/review',data=payload).status_code==403
     payload['csrf_token']=at
     assert admin.post(prefix+'/review',data={**payload,'csrf_token':'invalid'}).status_code==400
     assert admin.post(prefix+'/review',data=payload).status_code==302
     after=owner.get(prefix+'/analysis.json').get_json();assert after['original_prediction']==raw['original_prediction'];assert after['original_confidence']==raw['original_confidence'];assert after['final_class']=='Animal Sound';assert after['status']=='Reviewed'
     with app.app_context():
      event=db.session.get(AudioEvent,event_id);event.severity='Critical';db.session.commit()
     for action in ['acknowledge','escalate','dismiss']:
      assert admin.post(prefix+'/alert',data={'csrf_token':at,'action':action,'comment':'QA action'}).status_code==302
     actions=owner.get(prefix+'/analysis.json').get_json();assert actions['alert_state']=='Dismissed' and len(actions['actions'])==4
     assert owner.post('/workspace/profile',data={'full_name':'New Name'}).status_code==400
     assert owner.post('/workspace/profile',data={'csrf_token':ot,'full_name':'New Name'}).status_code==302
     assert owner.post('/workspace/settings',data={'csrf_token':ot}).status_code==403
     policy_path=root/'config/model_policy.json';policy_original=policy_path.read_bytes()
     try:
      r=admin.post('/workspace/settings',data={'csrf_token':at,'min_confidence':'65','min_margin':'15','alert_min_confidence':'85','alert_min_margin':'25'});assert r.status_code==302
      policy=json.loads(policy_path.read_text());assert policy['min_confidence']==.65 and not policy['validation_calibrated']
     finally:policy_path.write_bytes(policy_original)
     for route in [prefix,'/workspace/alerts','/workspace/manual-review?state=all','/workspace/event-history?q=test&min_confidence=0&max_confidence=100','/workspace/event-history?start=invalid','/workspace/event-history?audio_id=nope']:
      assert owner.get(route).status_code in [200,302],route
     with app.app_context():
      u=User.query.filter_by(email='owner@example.com').first()
      legacy=AudioEvent(user_id=u.id,original_filename='legacy.wav',stored_filename='missing.wav',sample_rate=22050,duration_seconds=3,quality_label='Good',rms=.2,predicted_class='Gunshot',confidence=.9,severity='Critical',status='Alert Generated');db.session.add(legacy);db.session.commit();legacy_id=legacy.id
     assert owner.get(f'/workspace/events/{legacy_id}').status_code==200
     assert owner.get(f'/workspace/events/{legacy_id}/audio').status_code==404
     assert admin.post(f'/workspace/events/{legacy_id}/alert',data={'csrf_token':at,'action':'acknowledge','comment':'QA legacy'}).status_code==302
     import soundfile as sf,numpy as np
     for name,arr in [('silence.wav',np.zeros(22050)),('short.wav',np.ones(2000)*.1),('long.wav',np.ones(22050*61)*.1)]:
      wav=io.BytesIO();sf.write(wav,arr,22050,format='WAV');wav.seek(0);assert owner.post('/api/audio/inspect',data={'audio':(wav,name)}).status_code==422,name
     print(json.dumps({'result':'PASS','checks':['all new pages render','real CNN upload with full scores and original metadata','exact duplicate warning','waveform and spectrogram data','audio, JSON and detail owner isolation','role checks and CSRF','review preserves original output','alert acknowledgement/escalation/dismissal history','legacy record compatibility','threshold settings update and restored','silence/short/long audio rejection']},indent=2))
