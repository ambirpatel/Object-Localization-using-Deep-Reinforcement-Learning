#%%
import os
import pickle
import math
import cv2
import numpy as np
from config import *

#%%
class State:
    cnn_model = load_model(os.path.join("vgg16.h5"))
    feature_extractor = K.function([cnn_model.layers[0].input], [cnn_model.layers[20].output])

    def __init__(self, history, bb, image):
        self.history = history
        self.bb = bb
        self.feature = State.compute_feature(history, bb, image)

    @staticmethod
    def compute_feature(history, bb, image):
        history_feature = State.get_history_feature(history)
        image_feature = State.get_image_feature(image, bb)
        feature = np.concatenate((image_feature, history_feature))
        return np.array([feature])

    @staticmethod
    def get_image_feature(image, bb):
        cropped = crop_image(bb, image)
        feature = State.feature_extractor([cropped.reshape(1, 224, 224, 3)])[0]
        return np.ndarray.flatten(feature)

    @staticmethod
    def get_history_feature(history):
        assert len(history) == history_length
        feature = np.zeros((90,))
        for i in range(history_length):
            action = history[i]
            if action != -1:
                feature[i * 9 + action] = 1
        return feature

#%%
def load_data(training_ratio):
    bbs = pickle.load(open(os.path.join("out_rl/", "bounding_boxes.p"), "rb"))
    print('loaded bbs')
    labels = pickle.load(open(os.path.join("out_rl/", "labels_rl.p"), "rb"))
    print('loaded labels')

    unique_indices = [i for i in range(len(labels)) if len(labels[i]) == 1]
    indices_to_load = unique_indices

    bbs = [bbs[i][0] for i in indices_to_load]
    labels = [labels[i] for i in indices_to_load]
    images = [cv2.imread(os.path.join("out_rl/imgs/", str(i) + ".png")) for i in indices_to_load]

    bbs_train = bbs[:int(len(bbs) * training_ratio)]
    bbs_test = bbs[int(len(bbs) * training_ratio):]
    labels_train = labels[:int(len(labels) * training_ratio)]
    labels_test = labels[int(len(labels) * training_ratio):]
    images_train = images[:int(len(images) * training_ratio)]
    images_test = images[int(len(images) * training_ratio):]

    return bbs_train, bbs_test, labels_train, labels_test, images_train, images_test, indices_to_load
#%%
def iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou
#%%
def crop_image(bb, image):
    w, h, d = image.shape
    bb = [int(math.floor(b)) for b in bb]
    bb[0] = max(bb[0], 0)
    bb[1] = max(bb[1], 0)
    bb[2] = min(bb[2], h)
    bb[3] = min(bb[3], w)
    cropped = image[bb[1]:bb[3], bb[0]:bb[2]]
    w, h, d = cropped.shape
    if w == 0 or h == 0:
        cropped = np.zeros((224, 224, 3))
    else:
        cropped = cv2.resize(cropped, (224, 224))
    return cropped
#%%
def get_features(images, bbs, labels):
    
    feature_to_all = []
    label = []

    for xi, yi, l, data_index in tqdm(zip(images, bbs, labels, range(len(images)))):
        (width, height, d) = xi.shape
        initial_history = [-1] * history_length
        initial_bb = (0, 0, height, width)
        s = State(initial_history, initial_bb, xi)
        feature = s.get_image_feature(xi, yi)
        feature_to_all.append(feature)
        label.append(l)
        
    return feature_to_all, label
#%%w