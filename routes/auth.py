from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from sqlalchemy import func
from extensions import db
from models_db import User

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter(func.lower(User.email) == email).first()
        if not user or not user.check_password(password):
            flash('Invalid email address or password.', 'danger')
            return render_template('auth/login.html', email=email)

        if not user.is_active:
            flash('Your account is inactive. Contact the administrator.', 'danger')
            return render_template('auth/login.html', email=email)

        session.clear()
        session['user_id'] = user.id
        session['user_name'] = user.full_name
        session['user_role'] = user.role
        session.permanent = remember

        flash(f'Welcome back, {user.full_name.split()[0]}!', 'success')
        next_url = request.args.get('next')
        return redirect(next_url or url_for('main.home'))

    return render_template('auth/login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = []
        if len(full_name) < 2:
            errors.append('Please enter your full name.')
        if '@' not in email or '.' not in email.rsplit('@', 1)[-1]:
            errors.append('Please enter a valid email address.')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters long.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if User.query.filter(func.lower(User.email) == email).first():
            errors.append('An account with this email already exists.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('auth/register.html', full_name=full_name, email=email)

        user = User(full_name=full_name, email=email, role='normal_user')
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Account created successfully. You can now sign in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')


@auth_bp.get('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'info')
    return redirect(url_for('main.home'))
