import numpy as np
import gym
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import random

img_rows , img_cols = 84, 84
#Convert image into Black and white
img_channels = 4 #We stack 4 frames


def test(model, val_samples, EVAL_STEPS, EVAL_EPSILON, ACTIONS, FRAME_PER_ACTION, render):
    env = gym.envs.make('Breakout-v0')
    env.reset()

    epsilon = EVAL_EPSILON

    total_reward = 0
    nrewards = 1
    nepisodes = 1
    episode_reward = 0
    max_reward = 0
    Q_total = 0
    num_QAs = 1

    Q_sa = 0
    action_index = 0
    r_t = 0
    a_t = np.zeros([ACTIONS])


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

    for steps in range(EVAL_STEPS):

        if steps % FRAME_PER_ACTION == 0:
            a_t = np.zeros([ACTIONS])
            if np.random.random() <= epsilon:
                #print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        x_t1_colored, r_t, terminal, info = env.step(np.argmax(a_t))
        episode_reward += r_t



        if(render=="True"):
           env.render()
        if(terminal):
            if(episode_reward > max_reward):
                max_reward = episode_reward
            total_reward += episode_reward
            nepisodes += 1
            episode_reward = 0
            env.reset()

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(img_rows, img_cols))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        s_t = s_t1

    avg_reward = total_reward/nepisodes

    Q_total = np.empty(val_samples.shape[0])

    for k, s in enumerate(val_samples):
        Q_total[k] = np.argmax(model.predict(s[0]))

    Q_avg = Q_total.mean()
    Q_total = Q_total.sum()
    return  total_reward, avg_reward, max_reward, Q_total, Q_avg
