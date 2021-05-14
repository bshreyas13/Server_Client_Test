# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:59:37 2021

@author: shrey
"""

import grpc
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import the generated classes
import StylizeImage_pb2
import StylizeImage_pb2_grpc

# data encoding
 
import base64
import time


# Function to load an image from a file, and add a batch dimension.
def load_img_(img):
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

def encode_toString(img):
    retval, buffer = cv2.imencode('.jpg',img)
    data = base64.b64encode(buffer)
    return data

def decode_toImage(data):
    
    data_ = base64.b64decode(data)
    img = np.frombuffer(data_, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


# open a gRPC channel
MAX_MESSAGE_LENGTH = 60000000
channel = grpc.insecure_channel('127.0.0.1:5005',
                                    options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])
# create a stub (client)
stub = StylizeImage_pb2_grpc.StylizeImageStub(channel)



t1 = time.time()
for _ in range(1000):

    # Load the input images.
    content_image = cv2.imread('content.jpg')
    style_image = cv2.imread('style.jpg')
    data = encode_toString(content_image)
    data_s = encode_toString(style_image)  
    # create a valid request message
    image_req = StylizeImage_pb2.B64Image(b64image_content = data,b64image_style = data_s)
    # make the call
    response = stub.Stylized_Image(image_req)
    out_img = decode_toImage(response.b64image)
    
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    imshow(content_image, 'Content Image')

    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 2)
    imshow(style_image, 'Style Image')

    # Visualize the output.
    plt.subplot(1, 3, 3)
    imshow(out_img, 'Stylized Image')
    
t2 = time.time()

print(t2-t1)