from flask import Blueprint, render_template, session, request, flash, redirect, url_for, abort, send_file, jsonify
from sqlalchemy import func
from extensions import db
from models_db import AudioEvent, User, EventAnalysis, EventAction
from services.event_display import (current_user, scoped_events, event_for_user, check_csrf, audio_path, recommendation, training_summary, REVIEW_ROLES, ALERT_ROLES)
from sqlalchemy.orm import selectinload
from utils.auth import login_required

workspace_bp = Blueprint('workspace', __name__, url_prefix='/workspace')


@workspace_bp.get('/live-monitoring')
@login_required
def live_monitoring():
    return render_template('workspace/live_monitoring.html')


@workspace_bp.get('/batch-upload')
@login_required
def batch_upload():
    return render_template(
        'workspace/batch_upload.html',
        title='Batch Audio Upload',
        eyebrow='MULTIPLE RECORDINGS',
        description='Upload several WAV, MP3, FLAC, OGG or M4A files and run the existing preprocessing and quality inspection pipeline on each file.'
    )


@workspace_bp.get('/overview')
@login_required
def overview():
    query = scoped_events()
    events = query.order_by(AudioEvent.created_at.desc()).limit(6).all()
    stats = {'total':query.count(), 'alerts':sum(e.alert_state in {'Open','Acknowledged','Escalated'} for e in query.filter(AudioEvent.severity.in_(['High','Critical'])).options(selectinload(AudioEvent.actions),selectinload(AudioEvent.analysis)).all()),
             'review':query.filter(AudioEvent.status.in_(['Manual Review','Uncertain'])).count(),
             'poor':query.filter(AudioEvent.quality_label.in_(['Poor','Unusable'])).count(),
             'disagree':query.filter(AudioEvent.agreement_status=='Model Disagreement').count(),
             'match':query.filter(AudioEvent.agreement_status.in_(['Acceptable Match','Weak Match'])).count()}
    return render_template('workspace/overview.html',events=events,stats=stats,admin=current_user().role in {'admin','administrator'})


@workspace_bp.get('/model-comparison')
@login_required
def model_comparison():
    selected,models = training_summary()
    key=request.args.get('model','custom_cnn')
    if key not in {'custom_cnn','random_forest','svm'}:key='custom_cnn'
    chosen=next(m for m in models if m['key']==key)
    return render_template('workspace/model_comparison.html',models=models,chosen=chosen,selected=selected)


@workspace_bp.get('/model-evidence/<model>/confusion.png')
@login_required
def model_confusion(model):
    if model not in {'custom_cnn','random_forest','svm'}:abort(404)
    from services.event_display import ROOT
    return send_file(ROOT/f'reports/model_evaluation/{model}_test_confusion_matrix.png',mimetype='image/png')


def history_query():
    query=scoped_events()
    args=request.args
    q=args.get('q','').strip()
    if q:query=query.filter(AudioEvent.original_filename.ilike(f'%{q}%'))
    audio_id=args.get('audio_id','').strip().lstrip('#')
    if audio_id:
        if not audio_id.isdigit():raise ValueError('Audio ID must be a number.')
        query=query.filter(AudioEvent.id==int(audio_id))
    for field in ['quality_label','predicted_class','severity','status','agreement_status']:
        value=args.get(field,'').strip()
        if value:query=query.filter(getattr(AudioEvent,field)==value)
    from datetime import datetime,timedelta
    for field,operator in [('start','start'),('end','end')]:
        if args.get(field):
            try:date=datetime.strptime(args[field],'%Y-%m-%d')
            except ValueError:raise ValueError('Choose a valid date range.')
            query=query.filter(AudioEvent.created_at>=date) if operator=='start' else query.filter(AudioEvent.created_at<date+timedelta(days=1))
    if args.get('start') and args.get('end') and args['start']>args['end']:raise ValueError('Start date must be before end date.')
    for key in ['min_confidence','max_confidence']:
        if args.get(key):
            try:value=float(args[key])/100
            except ValueError:raise ValueError('Confidence must be from 0 to 100.')
            if not 0<=value<=1:raise ValueError('Confidence must be from 0 to 100.')
            query=query.filter(AudioEvent.confidence>=value) if key=='min_confidence' else query.filter(AudioEvent.confidence<=value)
    if args.get('min_confidence') and args.get('max_confidence') and float(args['min_confidence'])>float(args['max_confidence']):raise ValueError('Minimum confidence must not exceed maximum confidence.')
    if current_user().role in {'admin','administrator'} and args.get('user_id'):
        if not args['user_id'].isdigit():raise ValueError('User ID must be a number.')
        query=query.filter(AudioEvent.user_id==int(args['user_id']))
    return query


@workspace_bp.get('/event-history')
@login_required
def event_history():
    from src.ml.training_config import CLASSES
    try:query=history_query()
    except ValueError as exc:
        flash(str(exc),'warning');return redirect(url_for('workspace.event_history'))
    pagination=query.options(selectinload(AudioEvent.actions),selectinload(AudioEvent.analysis)).order_by(AudioEvent.created_at.desc()).paginate(page=request.args.get('page',1,type=int),per_page=20,error_out=False)
    links={k:v for k,v in request.args.items() if k!='page'}
    return render_template('workspace/event_history.html',events=pagination.items,pagination=pagination,links=links,classes=CLASSES,agreements=['Acceptable Match','Weak Match','Model Disagreement','Uncertain Result','Python Only'],admin=current_user().role in {'admin','administrator'})


@workspace_bp.get('/alerts')
@login_required
def alerts():
    query=scoped_events().filter(AudioEvent.severity.in_(['High','Critical']))
    events=query.options(selectinload(AudioEvent.actions),selectinload(AudioEvent.analysis)).order_by(AudioEvent.created_at.desc()).limit(200).all()
    groups={'open':[],'pending':[],'handled':[],'other':[]}
    for e in events:
        state=e.alert_state
        if state=='Open' or state=='Escalated':groups['open'].append(e)
        elif state=='Pending review':groups['pending'].append(e)
        elif state in {'Acknowledged','Dismissed','Closed'}:groups['handled'].append(e)
        else:groups['other'].append(e)
    tab=request.args.get('tab','open')
    if tab not in groups:tab='open'
    return render_template('workspace/alerts.html',groups=groups,tab=tab,alerts=groups[tab],can_act=current_user().role in ALERT_ROLES)


@workspace_bp.get('/manual-review')
@login_required
def manual_review():
    from src.ml.training_config import CLASSES
    base=scoped_events().filter(AudioEvent.status.in_(['Uncertain','Manual Review','Reviewed']))
    counts={'pending':base.filter(AudioEvent.status!='Reviewed').count(),'reviewed':base.filter(AudioEvent.status=='Reviewed').count()}
    state=request.args.get('state','pending')
    query=base if state=='all' else (base.filter(AudioEvent.status=='Reviewed') if state=='reviewed' else base.filter(AudioEvent.status!='Reviewed'))
    events=query.options(selectinload(AudioEvent.actions),selectinload(AudioEvent.analysis)).order_by(AudioEvent.created_at.desc()).limit(100).all()
    return render_template('workspace/manual_review.html',review_items=events,counts=counts,state=state,classes=CLASSES,can_review=current_user().role in REVIEW_ROLES)


@workspace_bp.get('/events/<int:event_id>')
@login_required
def event_detail(event_id):
    from src.ml.training_config import CLASSES
    event=event_for_user(event_id)
    data=event.analysis.data if event.analysis else {}
    prediction=data.get('prediction') or {}
    return render_template('workspace/event_detail.html',event=event,data=data,prediction=prediction,classes=CLASSES,
                           recommended_action=recommendation(event.final_class,event.status),
                           can_review=current_user().role in REVIEW_ROLES,can_act=current_user().role in ALERT_ROLES)


@workspace_bp.get('/events/<int:event_id>/audio')
@login_required
def event_audio(event_id):
    event=event_for_user(event_id)
    return send_file(audio_path(event),conditional=True,download_name=event.original_filename)


@workspace_bp.get('/events/<int:event_id>/visuals')
@login_required
def event_visuals(event_id):
    from services.event_display import visual_data
    event=event_for_user(event_id)
    try:return jsonify(visual_data(audio_path(event)))
    except (ValueError,RuntimeError,OSError) as exc:return jsonify(error='Could not render this recording: '+str(exc)),422


@workspace_bp.get('/events/<int:event_id>/analysis.json')
@login_required
def event_json(event_id):
    event=event_for_user(event_id)
    payload={'audio_id':event.id,'filename':event.original_filename,'created_at_utc':event.created_at.isoformat(),
             'original_prediction':event.predicted_class,'original_confidence':event.confidence,'final_class':event.final_class,
             'status':event.status,'alert_state':event.alert_state,'analysis':event.analysis.data if event.analysis else None,
             'actions':[{'action':a.action,'actor_id':a.actor_id,'class':a.corrected_class,'comment':a.comment,'recommended_action':a.recommended_action,'time_utc':a.created_at.isoformat()} for a in event.actions]}
    response=jsonify(payload);response.headers['Content-Disposition']=f'attachment; filename=sonicsentinel-audio-{event.id}.json'
    return response


@workspace_bp.post('/events/<int:event_id>/review')
@login_required
def review_event(event_id):
    check_csrf()
    if current_user().role not in REVIEW_ROLES:abort(403)
    event=event_for_user(event_id)
    from src.ml.training_config import CLASSES
    label=request.form.get('corrected_class','');comment=request.form.get('comment','').strip();action=request.form.get('action','')
    if label not in CLASSES or action not in {'confirm','correct'} or not comment or len(comment)>2000:abort(400,description='Choose a class and provide a review comment (maximum 2,000 characters).')
    if action=='confirm' and label!=event.predicted_class:abort(400,description='Choose Correct when changing the original class.')
    advice=request.form.get('recommended_action','').strip()
    if len(advice)>1000:abort(400)
    if not event.analysis:
        db.session.add(EventAnalysis(event_id=event.id,data={'original_status':event.status,'original_severity':event.severity,'legacy_record':True}))
    db.session.add(EventAction(event_id=event.id,actor_id=current_user().id,action=action,corrected_class=label,comment=comment,recommended_action=advice))
    event.status='Reviewed';db.session.commit()
    flash('Review saved. Original model prediction and confidence are preserved.','success')
    if request.form.get('next')=='queue':return redirect(url_for('workspace.manual_review'))
    return redirect(url_for('workspace.event_detail',event_id=event.id))


@workspace_bp.post('/events/<int:event_id>/alert')
@login_required
def alert_action(event_id):
    check_csrf()
    if current_user().role not in ALERT_ROLES:abort(403)
    event=event_for_user(event_id);action=request.form.get('action','');comment=request.form.get('comment','').strip()
    if event.severity not in {'High','Critical'} or action not in {'acknowledge','dismiss','escalate'}:abort(400)
    if not comment or len(comment)>2000:abort(400,description='Add an action note (maximum 2,000 characters).')
    if not event.analysis:db.session.add(EventAnalysis(event_id=event.id,data={'original_status':event.status,'original_severity':event.severity,'legacy_record':True}))
    db.session.add(EventAction(event_id=event.id,actor_id=current_user().id,action=action,comment=comment))
    if action=='dismiss':event.status='Closed'
    db.session.commit();flash('Alert action recorded.','success')
    if request.form.get('next')=='alerts':return redirect(url_for('workspace.alerts',tab=request.form.get('tab','open')))
    return redirect(url_for('workspace.event_detail',event_id=event.id))


@workspace_bp.get('/reports')
@login_required
def reports():
    from services.reporting import report_data
    try:
        data = report_data(session['user_id'], request.args)
    except ValueError as exc:
        flash(str(exc), 'warning')
        return redirect(url_for('workspace.reports'))
    return render_template('workspace/reports.html', **data)


@workspace_bp.get('/reports/export.csv')
@login_required
def export_reports_csv():
    import csv
    import io
    from flask import Response
    from services.reporting import filtered_events
    try:
        query, *_ = filtered_events(session['user_id'], request.args)
    except ValueError as exc:
        return str(exc), 400
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['ID','Filename','Created UTC','Predicted class','Confidence','Severity','Status','Quality','Duration seconds'])
    def safe_cell(value):
        text = str(value if value is not None else '')
        return "'" + text if text.lstrip().startswith(('=','+','-','@')) or text.startswith(('\t','\r','\n')) else text
    for event in query.order_by(AudioEvent.created_at.desc()).yield_per(500):
        writer.writerow([safe_cell(v) for v in [event.id,event.original_filename,event.created_at.isoformat(),event.predicted_class,event.confidence,event.severity,event.status,event.quality_label,event.duration_seconds]])
    return Response('\ufeff'+output.getvalue(), mimetype='text/csv', headers={'Content-Disposition':'attachment; filename=sonicsentinel-events.csv'})


@workspace_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = db.session.get(User, session['user_id'])
    if request.method == 'POST':
        check_csrf()
        full_name = request.form.get('full_name', '').strip()
        if not 2 <= len(full_name) <= 120:
            flash('Please enter a valid full name.', 'danger')
        else:
            user.full_name = full_name
            db.session.commit()
            session['user_name'] = user.full_name
            flash('Profile updated successfully.', 'success')
            return redirect(url_for('workspace.profile'))
    return render_template('workspace/profile.html', user=user)


@workspace_bp.route('/settings',methods=['GET','POST'])
@login_required
def settings():
    from src.ml.reliability import read_policy
    from services.event_display import ROOT
    import json,os
    policy=read_policy(ROOT)
    admin=current_user().role in {'admin','administrator'}
    if request.method=='POST':
        check_csrf()
        if not admin:abort(403)
        try:
            updated=dict(policy)
            for key in ['min_confidence','min_margin','alert_min_confidence','alert_min_margin']:
                value=float(request.form[key])/100
                if not 0<=value<=1:raise ValueError('Thresholds must be from 0 to 100.')
                updated[key]=value
            if updated['alert_min_confidence']<updated['min_confidence'] or updated['alert_min_margin']<updated['min_margin']:
                raise ValueError('Alert thresholds must be at least as strict as classification thresholds.')
            cmp=dict(updated.get('comparison') or {})
            if request.form.get('python_weight','')!='':
                w=float(request.form['python_weight'])/100
                if not 0<=w<=1:raise ValueError('Python weight must be from 0 to 100.')
                cmp['fusion_weights']={'python':round(w,4),'gtm':round(1-w,4)}
            if request.form.get('acceptable_diff','')!='':
                d=float(request.form['acceptable_diff'])/100
                if not 0<=d<=1:raise ValueError('Confidence difference must be from 0 to 100.')
                cmp['acceptable_max_confidence_difference']=d
            cmp['review_on_model_disagreement']=request.form.get('review_on_model_disagreement')=='on'
            cmp['require_model_agreement_for_alert']=request.form.get('require_model_agreement_for_alert')=='on'
            updated['comparison']=cmp
            updated['validation_calibrated']=False
            from datetime import datetime
            updated['version']='ui-policy-'+datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
            updated['updated_by']=current_user().id
            updated['note']='Locally configured development rules; not validation-calibrated.'
            import tempfile
            path=ROOT/'config/model_policy.json'
            fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.json')
            try:
                with os.fdopen(fd,'w') as stream:json.dump(updated,stream,indent=2)
                os.replace(tmp,path)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
            flash('Rules saved for new analyses. Existing results remain unchanged.','success')
        except (ValueError,KeyError) as exc:flash(str(exc),'warning')
        return redirect(url_for('workspace.settings'))
    selected,models=training_summary()
    return render_template('workspace/settings.html',policy=policy,admin=admin,selected=selected)
