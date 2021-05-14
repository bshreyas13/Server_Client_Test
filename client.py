# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:59:37 2021

@author: shrey
"""

import grpc
import tensorflow as tf
import cv2
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
    # content_img = input('Path to content image:') 
    # style_img = input('Path to style image:')
    # compress
    # Load the input images.
    content_image = cv2.imread('content.jpg')
    style_image = cv2.imread('style.jpg')
    data = encode_toString(content_image)
    data_s = encode_toString(style_image)  
    # create a valid request message
    image_req = StylizeImage_pb2.B64Image(b64image_content = data,b64image_style = data_s)
    # make the call
    response = stub.Stylized_Image(image_req)
t2 = time.time()

print(t2-t1)