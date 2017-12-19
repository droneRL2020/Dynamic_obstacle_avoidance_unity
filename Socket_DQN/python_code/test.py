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
import os


# Unity connection
sio = socketio.Server()
app = Flask(__name__)

# Initial parameters

@sio.on('telemetry')
def telemetry(sid, data):
    Action = float(data["Action_vehicle"])
    # Current image input from the drone's camera
    imgString_front = data["front_image"]
    image_front = Image.open(BytesIO(base64.b64decode(imgString_front)))
    image_array_front = np.asarray(image_front)
    # Image transformation
    image_trans_front = cv2.resize(image_array_front, (80, 80))
    image_trans_front = cv2.cvtColor(image_trans_front, cv2.COLOR_RGB2GRAY)
    image_trans_front = np.reshape(image_trans_front, (80, 80, 1))

    print(data)
    send_control(1)

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
