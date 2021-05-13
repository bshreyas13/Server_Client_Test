# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:46:59 2021

@author: shrey
"""
  
import grpc
from concurrent import futures
import time

import Stylize_Image

# import the generated classes
import StylizeImage_pb2
import StylizeImage_pb2_grpc


# based on .proto service
class StylizeImageServicer(StylizeImage_pb2_grpc.ImageProcedureServicer):

    def ImageMeanWH(self, request, context):
        response = StylizeImage_pb2.Prediction()
        response.channel, response.mean  = Stylize_Image.predict(request.b64image_content,request.b64image_style)
        return response


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=12))


# add the defined class to the server
StylizeImage_pb2_grpc.add_StylizeImageServicer_to_server(
        StylizeImageServicer(), server)

# listen on port 5005
print('Starting server. Listening on port 5005.')
server.add_insecure_port('[::]:5005')
server.start()

try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    server.stop(0)