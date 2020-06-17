#!/usr/bin/python
import socket
from protocol_buffer_model.wrapper_hand_tracker_pb2 import WrapperHandTracking
import numpy as np
import os
import sys
import uuid
from pathlib import Path
from keras.models import load_model

UDP_IP = "127.0.0.1"
UDP_PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

handTrackingData = WrapperHandTracking()

def main():
    
    new_model = load_model('model/model.h5')
    new_model.summary()

    lines = []
    count = 0
    while count < 40:
        data, addr = sock.recvfrom(1024)
        handTrackingData.ParseFromString(data)
        if len(handTrackingData.landmarks.landmark) == 21 and handTrackingData.landmarks.landmark[0].x > 0.0:
            arr = []
            for i in range(len(handTrackingData.landmarks.landmark)):
                arr.append([])
                landmark = handTrackingData.landmarks.landmark[i]
                arr[i].append(landmark.x)
                arr[i].append(landmark.y)
                arr[i].append(landmark.z)
            lines.append(arr)
            count = count + 1
    input = np.array(lines)
    print(input.shape)
    yhat = new_model.predict(input.reshape(1, 40, 21, 3))
    print(yhat)

if __name__ == "__main__":
    main()
# {'turnonlights': array([0., 0., 0., 1.]), 
# 'turnonfan': array([0., 0., 1., 0.]), 
# 'turnofffan': array([1., 0., 0., 0.]), 
# 'turnofflights': array([0., 1., 0., 0.])}