from flask import Blueprint, url_for, redirect
from flask_login import LoginManager, login_required, logout_user

adminlogout = Blueprint('adminlogout', __name__, template_folder='../frontend')
login_manager = LoginManager()
login_manager.init_app(adminlogout)

@adminlogout.route('/adminlogout')
@login_required
def show():
    logout_user()
    return redirect(url_for('adminlogin.show') + '?success=logged-out')