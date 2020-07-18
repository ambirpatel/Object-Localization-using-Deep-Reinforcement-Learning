#%%
import random
from keras.layers import Dense
from keras import backend as K
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import cv2
import pickle
import math
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
from collections import deque
import collections
from tqdm import tqdm

from utils import *
from config import *

HUBER_DELTA = 1.0
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def initialize_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(4096 + 90,), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(9, activation='linear'))
    model.compile(loss=smoothL1, optimizer='adam')
    return model


loss_arr = []

def fit(model, x, y):
    global loss_arr
    loss = model.train_on_batch(x, y)
    loss_arr.append(loss)
    if len(loss_arr) == 100:
        print("loss %s" % str(sum(loss_arr) / len(loss_arr)))
        loss_arr = []


def transform(bb, a):

    alpha = .2
    alpha_w = alpha * (bb[2] - bb[0])
    alpha_h = alpha * (bb[3] - bb[1])
    dx1 = 0
    dy1 = 0
    dx2 = 0
    dy2 = 0

    if a == 0:
        dx1 = alpha_w
        dx2 = alpha_w
    elif a == 1:
        dx1 = -alpha_w
        dx2 = -alpha_w
    elif a == 2:
        dy1 = alpha_h
        dy2 = alpha_h
    elif a == 3:
        dy1 = -alpha_h
        dy2 = -alpha_h
    elif a == 4:
        dx1 = -alpha_w
        dx2 = alpha_w
        dy1 = -alpha_h
        dy2 = alpha_h
    elif a == 5:
        dx1 = alpha_w
        dx2 = -alpha_w
        dy1 = alpha_h
        dy2 = -alpha_h
    elif a == 6:
        dy1 = alpha_h
        dy2 = -alpha_h
    elif a == 7:
        dx1 = alpha_w
        dx2 = -alpha_w

    bb = (bb[0] + dx1, bb[1] + dy1, bb[2] + dx2, bb[3] + dy2)
    bb = (
        min(bb[0], bb[2]),
        min(bb[1], bb[3]),
        max(bb[0], bb[2]),
        max(bb[1], bb[3]),
    )

    return bb


def trigger_reward(bb, true_bb):
    return 3 if iou(bb, true_bb) > .6 else -3


def transform_reward(bb, bbp, true_bb):
    return 1 if iou(bbp, true_bb) > iou(bb, true_bb) else -1


def get_q(s, model):
    return np.ndarray.flatten(model.predict(s.feature))


def select_action(s, true_bb, step, epsilon, action_values):

    if step == max_steps:
        a = 8

    else:
        if random.random() > epsilon:
            a = np.argmax(action_values)

        else:

            action_rewards = [transform_reward(s.bb, transform(s.bb, a_tmp), true_bb) for a_tmp in range(8)]
            action_rewards.append(trigger_reward(s.bb, true_bb))
            action_rewards = np.array(action_rewards)
            positive_action_indices = np.where(action_rewards >= 0)[0]

            if len(positive_action_indices) == 0:
                positive_action_indices = list(range(0, 9))
            a = np.random.choice(positive_action_indices)


    return a


def take_action(s, true_bb, a, image):

    if a == 8:
        sp = s
        r = trigger_reward(s.bb, true_bb)
        took_trigger = True

    else:

        bb = s.bb
        bbp = transform(bb, a)
        r = transform_reward(bb, bbp, true_bb)
        took_trigger = False
        historyp = s.history[1:]
        historyp.append(a)
        assert len(historyp) == history_length
        sp = State(historyp, bbp, image)

    return sp, r, took_trigger


def weights_from_errors(errors):

    sorted_inds = sorted(range(len(errors)),key=lambda x: errors[x])
    inv_ranks = [0]*len(errors)

    for i in range(len(inv_ranks)):
        inv_ranks[sorted_inds[i]] = 1.0/(len(inv_ranks)-i)


    return inv_ranks


def apply_experience(main_model, target_model,experience, experience_errors):

    weights = weights_from_errors(experience_errors)
    sample_inds = random.choices(range(len(experience)), k=experience_sample_size, weights = weights)
    sample = [experience[i] for i in sample_inds]

    targets = np.zeros((experience_sample_size, 9))

    for i in range(experience_sample_size):
        s, a, r, sp, done = sample[i]
        target = r

        if not done:
            target = compute_target(r, sp, target_model)
        targets[i, :] = get_q(s, main_model)
        targets[i][a] = target

    x = np.concatenate([s.feature for (s, a, r, sp, d) in sample])
    fit(main_model, x, targets)


def compute_target(r, sp, target_model):
    return r + gamma * np.amax(get_q(sp, target_model))


def copy_main_to_target_model_weights(main_model, target_model):
    weights = main_model.get_weights()
    target_model.set_weights(weights)

def q_learning_train(x, y, labels, epochs, main_model, target_model):

    epsilon = epsilon_max
    experience = collections.deque(maxlen=experience_buffer_size)
    experience_errors = collections.deque(maxlen=experience_buffer_size)
    total_steps = 0

    for epoch in range(epochs):

        print("epoch %i" % epoch)

        for xi, yi, l, data_index in zip(x, y, labels, range(len(x))):

            (width, height, d) = xi.shape
            initial_history = [-1] * history_length
            initial_bb = (0, 0, height, width)
            s = State(initial_history, initial_bb, xi)
            done = False
            total_reward = 0
            step = 0

            while not done:

                action_values = get_q(s, main_model)
                a = select_action(s, yi, step, epsilon, action_values)
                sp, r, done = take_action(s, yi, a, xi)
                step_experience = (s, a, r, sp, done)

                #add the experience and td-error to our buffer
                experience.append(step_experience)
                experience_errors.append(abs(action_values[a]-compute_target(r,sp,target_model)))

                #apply the experience
                apply_experience(main_model, target_model, experience, experience_errors)
                s = sp
                total_reward += r
                step += 1
                total_steps += 1

                #update the target Q-network
                if total_steps % target_update_interval == 0:
                    copy_main_to_target_model_weights(main_model,target_model)

                # try:
                #     start_point = (s.bb[0], s.bb[2])
                #     print("start point {}".format(start_point))
                #     end_point = (s.bb[1], s.bb[3])
                #     print("end point {}".format(end_point))

                #     color = (255, 0, 0)
                #     thickness = 2
                    
                #     image = cv2.rectangle(xi, start_point, end_point, color, thickness)
                #     cv2.imshow('img', image)
                #     cv2.waitKey(10)
                # except:
                #     pass

            print("data_index %s" % data_index)
            print("reward %i" % total_reward)
            print("iou %f" % iou(s.bb, yi))

        if epoch < epsilon_dec_steps:
            epsilon -= epsilon_dec
            print("epsilon changed to %f" % epsilon)

    return main_model


def q_learning_predict(x,model):

    y = []
    count = 0
    for xi in x:

        (width, height, d) = xi.shape
        initial_history = [-1] * history_length
        initial_bb = (0, 0, height, width)
        s = State(initial_history, initial_bb, xi)

        # (width, height, d) = xi.shape
        # s = (0, 0, height, width)
        # history = [-1] * history_length
        done = False

        for i in range(sys.maxsize):

            action_values = get_q(s, model)
            if i == max_steps - 1:
                a = 8

            else:
                a = np.argmax(action_values)
            if a == 8:
                sp = s
                done = True

            else:
                bbp = transform(s.bb, a)
                historyp = s.history[1:]
                historyp.append(a)
                assert len(historyp) == history_length
                sp = State(historyp, bbp, xi)
                s = sp
            if done:
                break
        count+=1
        print("image ",count," predicted")
        
        # try:
        #     s.bb = [int(math.floor(b)) for b in s.bb]
        #     img = xi[s.bb[1]:s.bb[3], s.bb[0]:s.bb[2]]
        #     cv2.imshow('img', img)
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        #     print(s.bb)
        # except:
        #     pass

        y.append(s.bb)

    return y

def main():

    training_ratio = 1

    bbs_train, bbs_test, labels_train, labels_test, images_train, images_test, indices_to_load = load_data(training_ratio)

    print('images loaded')


    # features_csv, labels_csv = get_features(images_train, bbs_train, labels_train)
    # features_csv = pd.DataFrame(features_csv)
    # labels_csv = pd.DataFrame(labels_csv)
    # features_csv.to_csv('features.csv', index = False)
    # labels_csv.to_csv('lables.csv', index = False)

    if training:

        main_model = initialize_model()
        weights = main_model.get_weights()
        target_model = initialize_model()
        target_model.set_weights(weights)
        model = q_learning_train(images_train, bbs_train, labels_train, 15, main_model, target_model)
        model.save("dqn.h5")

    else:
        
        model = load_model("dqn.h5")
        y = q_learning_predict(images_test, model)
        inds = range(int(len(images_test) * training_ratio), len(images_test))

        np.savetxt("predicted_bounding_boxes.csv", y, delimiter=',', newline='\n')
        np.savetxt("predicted_image_indices.csv", inds, delimiter=',', newline='\n')
        np.savetxt("predicted_image_labels.csv", labels_test, delimiter=',', newline='\n')
#%%
main()
#%%