"""Local administrator utility; public registration never assigns privileged roles."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser()
parser.add_argument('--email',required=True)
parser.add_argument('--role',required=True,choices=['normal_user','audio_reviewer','security_operator','maintenance_operator','administrator'])
args=parser.parse_args()
from app import create_app
from extensions import db
from models_db import User
with create_app().app_context():
    user=User.query.filter_by(email=args.email.strip().lower()).first()
    if not user:raise SystemExit('User not found. Register the account in the app first.')
    user.role=args.role;db.session.commit();print('Role updated. Sign out and sign in again.')
