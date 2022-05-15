from os import listdir
import os
import cv2
import glob
from flask_monitoringdashboard.views.utils import *
import imagehash
import cv2
import imagehash
import numpy as np
from os import listdir
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn import tree, svm, linear_model
import pickle

des_list = []
true_positive = 1
true_negative = 1
false_positive = 1
false_negative = 1

im_contour_features = []

def renameImageForged():
    forged_images_path = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/createForged"
    x = os.path.join(forged_images_path, '*')
    x = glob.glob(x)
    for nameim in range(0, len(x)):
        img = cv2.imread(x[nameim])
        cv2.imwrite("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/forged/021001_00" + str(nameim) + ".png", img)


def createModel(userid):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    genuine_images_path = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/real"
    forged_images_path = "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/sign/forged"

    genuine_image_filenames = listdir(genuine_images_path)
    forged_image_filenames = listdir(forged_images_path)

    genuine_image_features = [[] for x in range(1)]
    forged_image_features = [[] for x in range(1)]

    for name in genuine_image_filenames:
        signature_id = int(name.split('_')[0][-3:])
        genuine_image_features[signature_id - 1].append({"name": name})

    for name in forged_image_filenames:
        signature_id = int(name.split('_')[0][-3:])
        forged_image_features[signature_id - 1].append({"name": name})

    for i in range(0, 1):
        z = i
        if i == 3:
            continue
        des_list = []
        for im in genuine_image_features[i]:
            image_path = genuine_images_path + "/" + im['name']
            # 1
            preprocessed_image = preprocess_image(image_path)
            # 2
            hash = imagehash.phash(Image.open(image_path))

            # 3
            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                get_contour_features(preprocessed_image.copy(), display=False)

            hash = int(str(hash), 16)
            im['hash'] = hash
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area

            im_contour_features.append(
                [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

            des_list.append(sift(preprocessed_image, image_path))

        for im in forged_image_features[i]:
            image_path = forged_images_path + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            hash = imagehash.phash(Image.open(image_path))

            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                get_contour_features(preprocessed_image.copy(), display=False)

            hash = int(str(hash), 16)
            im['hash'] = hash
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area

            im_contour_features.append(
                [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

            des_list.append(sift(preprocessed_image, image_path))

        # print(des_list)
        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))
        k = 100
        voc, variance = kmeans(descriptors, k, 1)

        # Calculate the histogram of features
        im_features = np.zeros((len(genuine_image_features[i]) + len(forged_image_features[i]), k + 4), "float32")

        for i in range(len(genuine_image_features[i]) + len(forged_image_features[i])):
            words, distance = vq(des_list[i][1], voc)
            for w in words:
                im_features[i][w] += 1

            for j in range(4):
                im_features[i][k + j] = im_contour_features[i][j]

        # Scaling the words
        stdSlr = StandardScaler().fit(im_features)
        im_features = stdSlr.transform(im_features)

        train_genuine_features, test_genuine_features = im_features[0:len(im_features) - 6], im_features[len(im_features) - 6:len(im_features) - 4]

        train_forged_features, test_forged_features = im_features[len(im_features) - 4:len(im_features) - 2], im_features[len(im_features) - 2:len(im_features)]

        # clf = linear_model.LogisticRegression(C=1e5)

        clf = LinearSVC(penalty="l1", loss='squared_hinge', dual=False)
        # clf = tree.DecisionTreeClassifier()
        # clf = tree.DecisionTreeRegressor()
        # clf = svm.SVC()
        clf.fit(np.concatenate((train_forged_features, train_genuine_features)),
                np.array(
                    [1 for x in range(len(train_forged_features))] + [2 for x in range(len(train_genuine_features))]))

        genuine_res = clf.predict(test_genuine_features)
        for res in genuine_res:
            if int(res) == 2:
                true_positive += 1
            else:
                false_negative += 1

        forged_res = clf.predict(test_forged_features)

        for res in forged_res:
            if int(res) == 1:
                true_negative += 1
            else:
                false_positive += 1

        print("true positive ",true_positive)
        print("false positive ",true_positive)
        print("true negative ",true_negative)
        print("false negative",false_negative)

        pickle.dump(clf, open("/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/models/model_" + str(userid) + ".pkl", "wb"))
    accuracy = float(true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
    precision = float(true_positive) / (true_positive + false_positive)
    recall = float(true_positive) / (true_positive + false_negative)
    f1_score = float(2 * precision * recall) / (precision + recall)

    print("Accuracy: ", round(accuracy, 2))
    print("Precision: ", round(precision, 2))
    print("Recall: ", round(recall, 2))
    print("F1 score: ", round(f1_score, 2))

def predictSignature(userid):

    model = pickle.load(open('/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/models/model_' + str(userid) + '.pkl', 'rb'))
    sig_path= "/home/mehdi/Bureau/Flask-MonitoringDashboard/flask_monitoringdashboard/static/images/signature.png"
    image = cv2.imread(sig_path)
    image = [[] for x in range(1)]
    image[0].append({"name": sig_path})
    des_list = []
    for im in image[0]:
        preprocessed_image = preprocess_image(sig_path)
        # 2
        hash = imagehash.phash(Image.open(sig_path))
        # 3
        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
            get_contour_features(preprocessed_image.copy(), display=False)
        hash = int(str(hash), 16)
        im['hash'] = hash
        im['aspect_ratio'] = aspect_ratio
        im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
        im['contour_area/bounding_area'] = contours_area / bounding_rect_area

        im_contour_features.append(
            [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

        des_list.append(sift(preprocessed_image, sig_path))
    return "forged"
    # print(des_list)
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    k = 100


    voc, variance = kmeans(descriptors, k, 1)
    # Calculate the histogram of features


    im_features = np.zeros((len(image), k + 4), "float32")


    for i in range(len(image)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

        for j in range(4):
            im_features[i][k + j] = im_contour_features[i][j]
    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    return model.predict(im_features)


