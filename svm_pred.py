#%%
import csv
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from collections import OrderedDict
from keras import backend as K
from keras.models import Sequential, load_model
import os
import cv2 
import numpy as np
from config import *
from utils import *
import pickle
from tqdm import tqdm
#%%
def load_data_1(training_ratio):
    bbs = pickle.load(open(os.path.join("out_rl_07/", "bounding_boxes.p"), "rb"))
    print('loaded bbs')
    labels = pickle.load(open(os.path.join("out_rl_07/", "labels_rl.p"), "rb"))
    print('loaded labels')

    unique_indices = [i for i in range(len(labels)) if len(labels[i]) == 1]
    indices_to_load = unique_indices

    bbs = [bbs[i][0] for i in indices_to_load]
    labels = [labels[i] for i in indices_to_load]
    images = [cv2.imread(os.path.join("out_rl_07/imgs/", str(i) + ".png")) for i in indices_to_load]

    bbs_train = bbs[:int(len(bbs) * training_ratio)]
    bbs_test = bbs[int(len(bbs) * training_ratio):]
    labels_train = labels[:int(len(labels) * training_ratio)]
    labels_test = labels[int(len(labels) * training_ratio):]
    images_train = images[:int(len(images) * training_ratio)]
    images_test = images[int(len(images) * training_ratio):]

    return bbs_train, bbs_test, labels_train, labels_test, images_train, images_test, indices_to_load

#%%
CLASSES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car",
           "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

#%%
training_ratio = 0

bbs_train, bbs_test, labels_train, labels_test, images_train, images_test, indices_to_load = load_data_1(training_ratio)
#%%
feature_train = []
with open('features.csv') as f:
    feature = csv.reader(f, delimiter=',')
    for row in feature:
        feature_train.append([float(i) for i in row])
#%%        
label_train = []
with open('lables.csv') as f:
    label = csv.reader(f, delimiter=',')
    for row in label:
        label_train.append([float(i) for i in row])
#%%
label_train = np.ndarray.flatten(np.array(label_train))
svm = LinearSVC()
svm.fit(feature_train, label_train)
print("fit completed")
#%%
predict_bbs = []
with open('predicted_bounding_boxes.csv') as f:
    bbs = csv.reader(f, delimiter=',')
    for row in bbs:
        predict_bbs.append([float(i) for i in row])
#%%
        
predict_feature = get_features(images_test, predict_bbs, labels_test)
#%%
ground_truth = get_features(images_test, bbs_test, labels_test)
#%%
label_DeepQ = svm.predict(predict_feature[0])
label_ground = svm.predict(ground_truth[0])


# accuracy_predict = OrderedDict()
# accuracy_ground = OrderedDict()
# sum_correct_predict = []
# sum_correct_ground = []
# #%%
# # labels_test = np.array(labels_test)#######    
# for i in range(20):
#     unique, counts = np.unique(labels_test, return_counts=True)
#     count = dict(zip(unique, counts))[i]
#     indexes = np.where(labels_test == i)[0]
#     unique1, counts1 = np.unique(label_DeepQ[indexes], return_counts=True)
#     count_predict = dict(zip(unique1, counts1))[i]
#     unique2, counts2 = np.unique(label_ground[indexes], return_counts=True)
#     count_ground = dict(zip(unique2, counts2))[i]
#     sum_correct_predict.append(count_predict)
#     sum_correct_ground.append(count_ground)
#     accuracy_predict[CLASSES[i]] = count_predict/count
#     accuracy_ground[CLASSES[i]] = count_ground/count   
    
# total_accu_predict = sum(sum_correct_predict)/len(labels_test)
# total_accu_ground = sum(sum_correct_ground)/len(labels_test)
# print('accuracy of predicted bounding boxes of test data in SVM:', accuracy_predict)
# print('accuracy of ground truth bounding boxes of test data in SVM:', accuracy_ground)
# print('accuracy of total correct classification for prediction = ', total_accu_predict)
# print('accuracy of total correct classification for ground truth = ', total_accu_ground)
# print("relative:")
# print([x/y for x,y in zip(accuracy_predict.values(), accuracy_ground.values())])
#%%