import flask
from flask import redirect, render_template, url_for, request, jsonify
from sqlalchemy.exc import IntegrityError

from flask_monitoringdashboard import blueprint, config
from flask_monitoringdashboard.core.auth import on_logout, on_login, secure, is_admin, admin_secure
from flask_monitoringdashboard.database import session_scope, User
from flask_monitoringdashboard.database.auth import get_user, get_all_users

from  flask_monitoringdashboard.views.model import *

MAIN_PAGE = config.blueprint_name + '.index'
BAD_REQUEST_STATUS = 400


@blueprint.route('/createaccount', methods=['GET', 'POST'])
def createaccount():
    if request.method == 'POST':
        login = request.form['login']
        password = request.form['password']
        password2 = request.form['password']
        lastname = request.form['lastname']
        firstname = request.form['firstname']
        mail = request.form['mail']
        with session_scope() as session:
            try:
                user = User(username=login, firstname=firstname, lastname=lastname, mail=mail)
                user.set_password(password=password)
                session.add(user)
                session.commit()
                user = get_user(username=login, password=password)
                print(user.id)
                createModel(user.id)

            except IntegrityError:
                return jsonify({'message': "Username already exists."}), BAD_REQUEST_STATUS
            except Exception as e:
                return jsonify({'message': str(e)}), BAD_REQUEST_STATUS


        return redirect(url_for(MAIN_PAGE))
    else:
        return render_template('fmd_register.html',
                               blueprint_name=config.blueprint_name)




