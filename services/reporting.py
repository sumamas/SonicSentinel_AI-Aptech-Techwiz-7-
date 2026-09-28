"""User-scoped stored-event analytics and immutable training benchmark summary."""
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import json
from sqlalchemy import func
from models_db import AudioEvent
from extensions import db

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#2f6fed','#f97316','#7c3aed','#16a34a','#f5c518','#06b6d4','#ec4899','#6366f1','#64748b','#0d9488','#94a3b8']
SEVERITIES = [('Critical','#ef4444'),('High','#f97316'),('Medium','#f5c518'),('Low','#22c55e'),('Informational','#38bdf8'),('Unclassified','#94a3b8')]

def filtered_events(user_id, args):
    today = datetime.utcnow().date()
    start_text = args.get('start') or (today-timedelta(days=29)).isoformat()
    end_text = args.get('end') or today.isoformat()
    try:
        start = datetime.strptime(start_text,'%Y-%m-%d')
        end = datetime.strptime(end_text,'%Y-%m-%d')
    except ValueError as exc:
        raise ValueError('Choose valid start and end dates.') from exc
    if start > end or (end-start).days > 365:
        raise ValueError('Choose a date range of 1 to 366 days, with start before end.')
    category = args.get('category','').strip()
    quality = args.get('quality','').strip()
    base = AudioEvent.query.filter_by(user_id=user_id)
    categories = [v[0] for v in db.session.query(AudioEvent.predicted_class).filter(AudioEvent.user_id==user_id,AudioEvent.predicted_class.isnot(None)).distinct().order_by(AudioEvent.predicted_class)]
    qualities = [v[0] for v in db.session.query(AudioEvent.quality_label).filter(AudioEvent.user_id==user_id).distinct().order_by(AudioEvent.quality_label)]
    query = base.filter(AudioEvent.created_at >= start,AudioEvent.created_at < end+timedelta(days=1))
    if category: query=query.filter(AudioEvent.predicted_class==category)
    if quality: query=query.filter(AudioEvent.quality_label==quality)
    return query, {'start':start_text,'end':end_text,'category':category,'quality':quality}, categories, qualities, start, end

def report_data(user_id,args):
    query,filters,categories,qualities,start,end=filtered_events(user_id,args)
    events=query.order_by(AudioEvent.created_at.desc()).all()
    total=len(events)
    confidence=[e.confidence for e in events if e.confidence is not None]
    cat=Counter(e.predicted_class or 'Unclassified' for e in events)
    quality=Counter(e.quality_label for e in events)
    sev=Counter(e.severity or 'Unclassified' for e in events)
    days=[(start+timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end-start).days+1)]
    daily=Counter(e.created_at.strftime('%Y-%m-%d') for e in events)
    high=Counter(e.created_at.strftime('%Y-%m-%d') for e in events if e.severity in ['Critical','High'])
    medium=Counter(e.created_at.strftime('%Y-%m-%d') for e in events if e.severity=='Medium')
    low=Counter(e.created_at.strftime('%Y-%m-%d') for e in events if e.severity in ['Low','Informational'])
    def rows(counter,colors):
        return [{'label':key,'count':count,'percent':round(count*100/total,1) if total else 0,'color':colors[i%len(colors)]} for i,(key,count) in enumerate(counter.most_common())]
    category_rows=rows(cat,COLORS)
    quality_rows=rows(quality,['#16a34a','#6ee7b7','#f5c518','#ef4444'])
    severity_rows=[{'label':label,'count':sev[label],'percent':round(sev[label]*100/total,1) if total else 0,'color':color} for label,color in SEVERITIES]
    selected=json.loads((ROOT/'models/selected_model.json').read_text())['selected_model']
    models=[]
    for key,name in [('svm','SVM'),('random_forest','Random Forest'),('custom_cnn','Custom CNN')]:
        path=ROOT/f'reports/model_evaluation/{key}_test_metrics.json'
        if path.is_file():
            data=json.loads(path.read_text())
            models.append({'key':key,'name':name,'selected':key==selected,'accuracy':data['accuracy'],'macro_f1':data['macro_f1'],'precision':data['macro_precision'],'recall':data['macro_recall']})
    summary={'total':total,'critical':sum(e.status=='Alert Generated' and e.severity=='Critical' for e in events),'average_confidence':sum(confidence)/len(confidence) if confidence else None,'review':sum(e.status in ['Manual Review','Uncertain'] for e in events)}
    return dict(filters=filters,categories=categories,qualities=qualities,summary=summary,category_rows=category_rows,quality_rows=quality_rows,severity_rows=severity_rows,models=models,selected_model=selected,recent_events=events[:20],chart_data={'days':days,'total':[daily[d] for d in days],'high':[high[d] for d in days],'medium':[medium[d] for d in days],'low':[low[d] for d in days],'categories':category_rows,'quality':quality_rows})
