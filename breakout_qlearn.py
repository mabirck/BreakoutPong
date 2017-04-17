#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
import random
import numpy as np
from collections import deque
import keras

import json
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam, RMSprop
import tensorflow as tf

import gym
import test

GAME = 'breakout' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
EVAL_EPSILON = 0.05
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 0.99 # starting value of epsilon
REPLAY_MEMORY = 1000000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 4
LEARNING_RATE = 1e-4
TOTAL = 10000000
SAVE_MODEL = 50000
EPOCH_LENGTH = 50016

EVAL_STEPS = 10000

img_rows , img_cols = 84, 84
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(16, (8,8), strides=(4, 4), padding='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    #model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    #model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256, activity_regularizer=keras.regularizers.l1_l2(0.42)))
    model.add(Activation('relu'))
    model.add(Dense(6, activity_regularizer=keras.regularizers.l1_l2(0.42) ))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=RMSprop())
    print("We finish building the model")
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    #game_state = game.GameState()
    render = args['render']
    env = gym.envs.make("Breakout-v0")
    env.reset()
    if(render == "True"):
       env.render()
    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, info = env.step(np.argmax(do_nothing))
    if(render == "True"):
       env.render()
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(img_rows, img_cols))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,1))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    val_samples = None

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    loss = 0
    batch_count = 0


    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])

        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                #print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal, info = env.step(np.argmax(a_t))

        if(render=="True"):
           env.render()
        if(terminal):
            env.reset()
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(img_rows, img_cols))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:

            if val_samples == None:
                val_samples = random.sample(D, BATCH)
                val_samples = np.asarray(val_samples)

            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)



            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            #print (inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)


                ###### METRICS TO EVALUATE ##############

            # targets2 = normalize(targets)

            batch_count+=32
            loss += model.train_on_batch(inputs, targets)

            if(batch_count % EPOCH_LENGTH == 0 and t >= OBSERVE):

                total_reward, avg_reward, max_reward, Q_total, Q_avg = test.test(model, val_samples, EVAL_STEPS, EVAL_EPSILON, ACTIONS, FRAME_PER_ACTION,render)

                print("EPOCH", batch_count/EPOCH_LENGTH, "/ STATE", state, \
                    "/ EPSILON", epsilon, "/ REWARD", total_reward,  "/ MAX REWARD", max_reward, \
                     "/AVG REWARD", avg_reward, "/ Q_total " , Q_total,  "/ Q_avg", Q_avg , "/ Loss ", loss/(EPOCH_LENGTH))

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % SAVE_MODEL == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        #print("TIMESTEP", t, "/ STATE", state, \
        #    "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
        #    "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
        if(t > TOTAL):
            break

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-r','--render',help='Render the game', required=False, default=False)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
