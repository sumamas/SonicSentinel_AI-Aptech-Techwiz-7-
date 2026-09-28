from functools import wraps
from flask import session, redirect, url_for, request, flash


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please sign in to continue.', 'warning')
            return redirect(url_for('auth.login', next=request.path))
        return view(*args, **kwargs)
    return wrapped_view
