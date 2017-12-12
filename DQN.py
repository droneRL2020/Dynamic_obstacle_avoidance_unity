# Basic Communication Code

# Modules for unity

import argparse
import base64
import json

import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

# Modules for DQN
import tensorflow as tf
import math 
import cv2
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime

# Unity connection
sio = socketio.Server()
app = Flask(__name__)

#DQN Parameters
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025
First_epsilon = 1.0
Final_epsilon = 0.01
Epsilon = First_epsilon

Num_replay_memory = 500
Num_start_training = 250
Num_training = 1000
Num_update = 200
Num_batch = 32
Num_skipFrame = 4
Num_stackFrame = 4
Num_colorChannel = 1
Num_MapChannel = 1

img_size = 80

first_conv_img = [8,8, Num_colorChannel * Num_stackFrame * 2, 32]
second_conv = [4,4,32,64]
third_conv = [3,3,64,64]
first_dense_img = [10*10*64, 1024]
first_dense = [10*10*64, 512]
second_dense = [512, 256]
third_dense = [256, Num_action]

# Initialize weights and bias
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))
def bias_variable(shape):
    return tf.Variable(xavier_initializer(shape))
# Xavier Weights initializer
def xavier_initializer(shape):
    dim_sum = np.sum(shape)
    if len(shape) == 1:
        dim_sum += 1
    bound = np.sqrt(2.0 / dim_sum)
    return tf.random_uniform(shape, minval=-bound, maxval=bound)
# Convolution and pooling
def conv2d(x,w,stride):
    return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def assign_network_to_target():
    update_wconv1_img = tf.assign(w_conv1_target_img, w_conv1_img)
    update_wconv2_img = tf.assign(w_conv2_target_img, w_conv2_img)
    update_wconv3_img = tf.assign(w_conv3_target_img, w_conv3_img)
    update_bconv1_img = tf.assign(b_conv1_target_img, b_conv1_img)
    update_bconv2_img = tf.assign(b_conv2_target_img, b_conv2_img)
    update_bconv3_img = tf.assign(b_conv3_target_img, b_conv3_img)

    sess.run(update_wconv1_img)
    sess.run(update_wconv2_img)
    sess.run(update_wconv3_img)
    sess.run(update_bconv1_img)
    sess.run(update_bconv2_img)
    sess.run(update_bconv3_img)
    
# Input
x_img = tf.placeholder(tf.float32, shape = [None, img_size, img_size, 2 * Num_colorChannel * Num_stackFrame])
# Normalize input
x_img = (x_img - (255.0/2)) / (255.0/2)

###################################### Image Network ######################################
# Convolution variables
w_conv1_img = weight_variable(first_conv_img)
b_conv1_img = bias_variable([first_conv_img[3]])
w_conv2_img = weight_variable(second_conv)
b_conv2_img = bias_variable([second_conv[3]])
w_conv3_img = weight_variable(third_conv)
b_conv3_img = bias_variable([third_conv[3]])

# Densely connect layer variables
w_fc1 = weight_variable(first_dense)
b_fc1 = bias_variable([first_dense[1]])
w_fc2 = weight_variable(second_dense)
b_fc2 = bias_variable([second_dense[1]])
w_fc3 = weight_variable(third_dense)
b_fc3 = bias_variable([third_dense[1]])                             

# Network
h_conv1_img = tf.nn.relu(conv2d(x_img, w_conv1_img, 4) + b_conv1_img)
h_conv2_img = tf.nn.relu(conv2d(h_conv1_img, w_conv2_img, 2) + b_conv2_img)
h_conv3_img = tf.nn.relu(conv2d(h_conv2_img, w_conv3_img, 1) + b_conv3_img)
h_pool3_flat_img = tf.reshape(h_conv3_img, [-1, first_dense_img[0]])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat_img, w_fc1)+b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)
output = tf.matmul(h_fc2, w_fc3) + b_fc3


############################# Image Target Network ###############################
# Convolution variables target
w_conv1_target_img = weight_variable(first_conv_img)
b_conv1_target_img = bias_variable([first_conv_img[3]])
w_conv2_target_img = weight_variable(second_conv)
b_conv2_target_img = bias_variable([second_conv[3]])
w_conv3_target_img = weight_variable(third_conv)
b_conv3_target_img = bias_variable([third_conv[3]])

# Densely connect layer variables target
w_fc1_target = weight_variable(first_dense)
b_fc1_target = bias_variable([first_dense[1]])
w_fc2_target = weight_variable(second_dense)
b_fc2_target = bias_variable([second_dense[1]])
w_fc3_target = weight_variable(third_dense)
b_fc3_target = bias_variable([third_dense[1]])

# Target Network
h_conv1_target_img = tf.nn.relu(conv2d(x_img, w_conv1_target_img, 4) + b_conv1_target_img)
h_conv2_target_img = tf.nn.relu(conv2d(h_conv1_target_img, w_conv2_target_img, 2) + b_conv2_target_img)
h_conv3_target_img = tf.nn.relu(conv2d(h_conv2_target_img, w_conv3_target_img, 1) + b_conv3_target_img)
h_pool3_flat_target_img = tf.reshape(h_conv3_target_img, [-1, first_dense_img[0]])
h_fc1_target = tf.nn.relu(tf.matmul(h_pool3_flat_target_img, w_fc1_target)+b_fc1_target)
h_fc2_target = tf.nn.relu(tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target)
output_target = tf.matmul(h_fc2_target, w_fc3_target) + b_fc3_target

############################# Calculate Loss & Train #################################

# Loss function and Train
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_prediction = tf.placeholder(tf.float32, shape = [None])
y_target = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
train_step = tf.train.AdamOptimizer(learning_rate = Learning_rate, epsilon = 1e-02).minimize(Loss)


# Initialize variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.InteractiveSession(config=config)
init = tf.global_variables_initializer()
sess.run(init)

# Load the file if the saved file exists
saver = tf.train.Saver()
# check_save = 1
check_save = input('Is there any saved data?(1=y/2=n): ')


if check_save == 1:
    checkpoint = tf.train.get_checkpoint_state("./saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

# Initial parameters
Replay_memory = []
step = 1
Init = 0
state = 'Observing'
episode = 0
img_front_old = 0
observation_set_img = []
action_old =  np.array([1,0,0,0,0])

# date - hour - minute of training time
date_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)


@sio.on('telemetry')
def telemetry(sid, data):
    global step, Replay_memory, Epsilon, img_front_old, observation_set_img
    current_time = time.time()
    # Initialization
    if Init == 0:
        observation_next_img = np.zeros([img_size, img_size, 2])
        observation_in_img = np.zeros([img_size, img_size, 1])
    
    
    # Get data from Unity
    movement = float(data["Movement"])   
    send_control(action)

# Connection with Unity
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(-1)


# Send control to Unity
def send_control(action):
    sio.emit("onsteer", data={
        'action': action.__str__()
        # 'num_connection': num_connection.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)