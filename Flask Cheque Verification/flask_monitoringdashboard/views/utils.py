##################################
import tensorflow as tf
import logging
import urllib.request
import os
from twilio.rest import Client
import io
import numpy as np
from six import BytesIO
from PIL import Image
import json
import cv2

from keras.models import model_from_json
from skimage.filters import threshold_otsu

logger = logging.getLogger(__name__)

import math


def ExtractAmount(openTest,x,y,w,h):
    return (openTest[y:y+h, x:x+w])

def NameSegmentation():
    y = 190
    x = 320
    h = 135
    w = 1400
    #legal_amount = ExtractAmount(imageTest, x, y, w, h)
    #cv2.imwrite('PreprocessingFinal/legal_amount.png', legal_amount)

#Generate Forged
def verticalImage(img, number):
    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
            offset_y = 0
            if j + offset_x < rows:
                img_output[i, j] = img[i, (j + offset_x) % cols]
            else:
                img_output[i, j] = 255
    cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/forged/021001_000.png", img_output)

#####################
# Horizontal wave
def horizontalImage(img, number):
    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = 0
            offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
            if i + offset_y < rows:
                img_output[i, j] = img[(i + offset_y) % rows, j]
            else:
                img_output[i, j] = 255
    cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/forged/021001_001.png", img_output)


#####################
# Both horizontal and vertical
def bothImage(img, number):
    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
            offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
            if i + offset_y < rows and j + offset_x < cols:
                img_output[i, j] = img[(i + offset_y) % rows, (j + offset_x) % cols]
            else:
                img_output[i, j] = 255
    cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/forged/021001_002.png", img_output)


#####################
# Concave effect
def concaveImage(img, number):
    rows, cols = img.shape
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2 * cols)))
            offset_y = 0
            if j + offset_x < cols:
                img_output[i, j] = img[i, (j + offset_x) % cols]
            else:
                img_output[i, j] = 255
    cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/forged/021001_003.png", img_output)

def createForged(imgName,i):
    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    verticalImage(img,i)
    horizontalImage(img,i)
    bothImage(img,i)
    concaveImage(img,i)

def threshold_image(img_arr):
    thresh = threshold_otsu(img_arr)
    return np.where(img_arr > thresh, 255, 0)


def resize_img(image_np, img_size=(224, 224)):
    image_np = Image.fromarray(image_np)
    return np.array(image_np.resize(img_size, Image.BILINEAR))


def resnet_preprocess(image_np, resize_input=True, threshold_input=True, invert_input=True, resnet=True):
    if invert_input:
        image_np = invert_img(image_np)
    if resize_input:
        image_np = resize_img(image_np)
    if threshold_input:
        image_np = threshold_image(image_np)
    if resnet:
        image_np = tf.keras.applications.resnet.preprocess_input(image_np)
    return image_np


def mkdir(dir_path: str, exist_ok: bool=True) -> None:
    os.makedirs(dir_path, exist_ok=exist_ok)


def download_file(file_url: str, file_name: str,  destination_dir: str) -> str:
    mkdir(destination_dir)

    # add header agent as needed
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    logger.info(">> Downloading data file for " + file_url)
    file_path = os.path.join(destination_dir, file_name)
    urllib.request.urlretrieve(file_url, file_path)
    return file_path


def read_file(file_path: str):
    return tf.io.gfile.GFile(file_path, "rb").read()


def invert_img(img):
    return np.invert(img)


def img_to_np_array(img_path: str, invert_image=False) -> None:
    img = read_file(img_path)
    image = Image.open(BytesIO(img)).convert('RGB')
    img_np = tf.keras.preprocessing.image.img_to_array(image).astype(np.uint8)
    # np.array(image.getdata()).reshape(
    #     (im_height, im_width, 3)).astype(np.uint8)
    if invert_image:
        img_np = invert_img(img_np)

    return img_np


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


def save_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_model_from_weights(model_dir):
    model = model_from_json(load_json_file(
        os.path.join(model_dir, "model_architecture.json")))
    model.load_weights(os.path.join(model_dir, "model_weights.h5"))
    return model


class Cleaner():
    def __init__(self, model_type="unet", batch_size=64):
        self.model_type = model_type
        self.batch_size = batch_size

    def load(self, model_path: str):
        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"PSNR": None, "SSIM": None})

    def clean(self, image_np):
        return self.model.predict(image_np, batch_size=self.batch_size)

def remove_noise(image):
    return cv2.medianBlur(image,5)

def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def seg(image):
    ret,thresh = cv2.threshold(image,170,255,cv2.THRESH_BINARY_INV)
    #Image Dilation
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=4)
    #Segmentation Part
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    return(image)

def verify(imagePath,seuil):
    img = cv2.imread(imagePath)
    number_of_black_pix = np.sum (img == 0)
    if(number_of_black_pix>seuil):
        return("Not Empty")
    else:
        return("Empty")

def namePreprocessing(namePath):
    y = 190
    x = 320
    h = 135
    w = 1400
    image = cv2.imread(namePath)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, image_black = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    nv_image = remove_noise(image_black)
    # cv2.imwrite('dhia/black_and_white/image_black'+str(i)+'.png', erode(nv_image))
    img = ExtractAmount(nv_image, x, y, w, h)
    img = erode(img)
    # img = cv2.imread(amounts_path[i])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(img, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
    rho, theta, thresh = 2, np.pi / 180, 400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)
    if lines is None:
        cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/name/name.png", seg(img))
    elif lines[0][0][0] > 125.0 or lines[0][0][0] < 20.0:
        for rho, theta in lines[1]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1600 * (-b))
            y1 = int(y0 + 1600 * (a))
            x2 = int(x0 - 1600 * (-b))
            y2 = int(y0 - 1600 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    else:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1600 * (-b))
            y1 = int(y0 + 1600 * (a))
            x2 = int(x0 - 1600 * (-b))
            y2 = int(y0 - 1600 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/name/name.png", seg(img))

def amountPreprocessing(imagePath):
    y = 390
    x = 1780
    h = 160
    w = 515
    image = cv2.imread(imagePath)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, image_black = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    nv_image = remove_noise(image_black)
    # cv2.imwrite('dhia/black_and_white/image_black'+str(i)+'.png', erode(nv_image))
    img = ExtractAmount(nv_image, x, y, w, h)
    img = erode(img)
    # img = cv2.imread(amounts_path[i])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(img, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
    rho, theta, thresh = 2, np.pi / 180, 400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)
    if lines is None:
        cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/amount/amount.png", seg(img))
    elif lines[0][0][0] > 125.0 or lines[0][0][0] < 20.0:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1600 * (-b))
            y1 = int(y0 + 1600 * (a))
            x2 = int(x0 - 1600 * (-b))
            y2 = int(y0 - 1600 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    else:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1600 * (-b))
            y1 = int(y0 + 1600 * (a))
            x2 = int(x0 - 1600 * (-b))
            y2 = int(y0 - 1600 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/amount/amount.png", seg(img))

def datePreprocessing(imagePath):
    y = 75
    x = 1780
    h = 85
    w = 515
    image = cv2.imread(imagePath)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, image_black = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
    nv_image = remove_noise(image_black)
    # cv2.imwrite('dhia/black_and_white/image_black'+str(i)+'.png', erode(nv_image))
    img = ExtractAmount(nv_image, x, y, w, h)
    img = erode(img)
    # img = cv2.imread(amounts_path[i])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(img, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
    rho, theta, thresh = 2, np.pi / 180, 400
    lines = cv2.HoughLines(bin_img, rho, theta, thresh)
    if lines is None:
        cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/date/date.png", seg(img))
    elif lines[0][0][0] > 125.0 or lines[0][0][0] < 20.0:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1600 * (-b))
            y1 = int(y0 + 1600 * (a))
            x2 = int(x0 - 1600 * (-b))
            y2 = int(y0 - 1600 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    else:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1600 * (-b))
            y1 = int(y0 + 1600 * (a))
            x2 = int(x0 - 1600 * (-b))
            y2 = int(y0 - 1600 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)
    cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/date/date.png", seg(img))


def get_contour_features(preprocessed_image, display=False):
    """
    :param preprocessed_image: preprocessed image
    :param display: flag - if true display images
    :return: aspect ratio of bounding rectangle, area of bounding rectangle, contours and convex hull
    """

    rect = cv2.minAreaRect(cv2.findNonZero(preprocessed_image))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])

    aspect_ratio = max(w, h) / min(w, h)
    bounding_rect_area = w * h

    if display:
        image1 = cv2.drawContours(preprocessed_image.copy(), [box], 0, (120, 120, 120), 2)

    hull = cv2.convexHull(cv2.findNonZero(preprocessed_image))

    if display:
        convex_hull_image = cv2.drawContours(preprocessed_image.copy(), [hull], 0, (120, 120, 120), 2)

    contours, hierarchy = cv2.findContours(preprocessed_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        contour_image = cv2.drawContours(preprocessed_image.copy(), contours, -1, (120, 120, 120), 3)

    contour_area = 0
    for cnt in contours:
        contour_area += cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)

    return aspect_ratio, bounding_rect_area, hull_area, contour_area

def preprocess_image(image_path, display=False):
    raw_image = cv2.imread(image_path)
    bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    bw_image = 255 - bw_image
    _, threshold_image = cv2.threshold(bw_image, 30, 255, 0)
    return threshold_image

def sift(preprocessed_image, image_path, display=False):
    raw_image = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(preprocessed_image, None)

    if display:
        cv2.drawKeypoints(preprocessed_image, kp, raw_image)


    return (image_path, des)
def send_sms(to):
    #account_sid = os.environ['ACe89302e72f47f696134999d4049acdec']
    #auth_token = os.environ['1e223374739469f1dca9adb5fc62a8c0']
    client = Client('ACe89302e72f47f696134999d4049acdec', '1e223374739469f1dca9adb5fc62a8c0')

    message = client.messages.create(
             body='the cheque is valid  ',
             from_='+19789612940',
             to=to
         )
    return 'sent'

def send_sms_failed(to):
    #account_sid = os.environ['ACe89302e72f47f696134999d4049acdec']
    #auth_token = os.environ['1e223374739469f1dca9adb5fc62a8c0']
    client = Client('ACe89302e72f47f696134999d4049acdec', '1e223374739469f1dca9adb5fc62a8c0')

    message = client.messages.create(
             body='the cheque is not valid  ',
             from_='+19789612940',
             to=to
         )
    return 'sent'