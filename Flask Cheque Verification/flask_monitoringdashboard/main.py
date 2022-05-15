"""This file can be executed for developing purposes.
To run use:

Note: This is not used when the flask_monitoring_dashboard
is attached to your flask application.
"""


import time
from random import random, randint

from flask import Flask, redirect, url_for,request,flash

import flask_monitoringdashboard as dashboard
import re
from flask_monitoringdashboard.views.utils import *
from flask_monitoringdashboard.views.model import *
import base64
import pymysql

app = Flask(__name__)
dashboard.config.version = '3.2'
dashboard.config.group_by = '2'
#dashboard.config.database_name = 'sqlite:///data.db'
dashboard.config.database_name = 'mysql+pymysql://root:@localhost:3306/database'
# dashboard.config.database_name = 'postgresql://user:password@localhost:5432/mydb'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def on_the_minute():
    return int(random() * 100 // 10)


minute_schedule = {'second': 00}
dashboard.add_graph("On Half Minute", on_the_minute, "cron", **minute_schedule)


def every_ten_seconds():
    return int(random() * 100 // 10)


every_ten_seconds_schedule = {'seconds': 10}
dashboard.add_graph("Every 10 Seconds", every_ten_seconds, "interval", **every_ten_seconds_schedule)
dashboard.bind(app)


@app.route('/')
def to_dashboard():

    return redirect(url_for(dashboard.config.blueprint_name + '.login'))


@app.route('/createaccount',methods=['POST','GET'])
def to_create():
    return redirect(url_for(dashboard.config.blueprint_name + '.createaccount'))

@app.route('/test', methods=['GET', 'POST'])
def saveImage():

    image1 = request.form['image1']
    image2 = request.form['image2']
    image3 = request.form['image3']
    image4 = request.form['image4']


    header, encoded = image1.split(",", 1)
    data = base64.b64decode(encoded)
    with open("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/real/001001_000.png", "wb") as f:
        f.write(data)

    header, encoded = image2.split(",", 1)
    data = base64.b64decode(encoded)
    with open("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/real/001001_001.png", "wb") as f:
        f.write(data)

    header, encoded = image3.split(",", 1)
    data = base64.b64decode(encoded)
    with open("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/real/001001_002.png", "wb") as f:
        f.write(data)

    header, encoded = image4.split(",", 1)
    data = base64.b64decode(encoded)
    with open("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/real/001001_003.png", "wb") as f:
        f.write(data)
    createForged("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/real/001001_000.png", "10")

    return redirect(url_for(dashboard.config.blueprint_name + '.login'))

@app.route('/', methods=['POST'])
def predict():

    #import cleaner
    cleaner_model_path = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/signver/models/cleaner/small"
    cleaner = Cleaner()
    cleaner.load(cleaner_model_path)

    imagefile = request.files['imagefile']
    print(imagefile)
    if imagefile.filename == '':
        flash('No image selected for uploading')
        return redirect(url_for(dashboard.config.blueprint_name + '.login'))

    if imagefile and allowed_file(imagefile.filename):
        image_path = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/" + imagefile.filename
        imagefile.save(image_path)
        test_line = img_to_np_array(image_path)
        #Cheque Segmentation and Applying preprocessing
        test_line = resnet_preprocess(test_line[650:1000, 1650:], resnet=False, invert_input=False)

        static = cv2.imread(image_path)

        cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/cheque.png",static)

        cn = cleaner.clean(np.expand_dims(test_line, axis=0))
        final_image = cn.astype(np.int64) * 255
        cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/signature.png", final_image.reshape(224, 224, 3))
        signpath="/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/signature.png"

        namePreprocessing(image_path)
        amountPreprocessing(image_path)
        datePreprocessing(image_path)

        namepath = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/name/name.png"
        if verify(namepath,17000) == "Empty":
            send_sms_failed("+21627055806")


        amountpath = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/amount/amount.png"
        if verify(amountpath, 7000) == "Empty":
            send_sms_failed("+21627055806")

        datePath = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/date/date.png"
        if verify(datePath, 5500) == "Empty":
            send_sms_failed("+21627055806")
        print(predictSignature(39))
        flash('Image successfully uploaded and displayed below')
        return redirect(url_for(dashboard.config.blueprint_name + '.login'))
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif, tif')
        return redirect(url_for(dashboard.config.blueprint_name + '.login'))
    return redirect(url_for(dashboard.config.blueprint_name + '.login'))


@app.route('/endpoint')
def endpoint():
    # if session_scope is imported at the top of the file, the database config won't take effect
    from flask_monitoringdashboard.database import session_scope

    with session_scope() as session:
        print(session.bind.dialect.name)

    print("Hello, world")
    return 'Ok'


@app.route('/endpoint2')
def endpoint2():
    time.sleep(0.5)
    return 'Ok', 400


@app.route('/endpoint3')
def endpoint3():
    if randint(0, 1) == 0:
        time.sleep(0.1)
    else:
        time.sleep(0.2)
    return 'Ok'


@app.route('/endpoint4')
def endpoint4():
    time.sleep(0.5)
    return 'Ok'


@app.route('/endpoint5')
def endpoint5():
    time.sleep(0.2)
    return 'Ok'


def my_func():
    # here should be something actually useful
    return 33.3


if __name__ == "__main__":
    app.run()
