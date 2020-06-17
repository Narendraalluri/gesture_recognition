#!/usr/bin/python
import socket
from protocol_buffer_model.wrapper_hand_tracker_pb2 import WrapperHandTracking
import numpy as np
import os
import sys
import uuid
from pathlib import Path
from keras.models import load_model
import time
import pickle 

pkl_file = open('model/image_label_encoder.pkl', 'rb')
le_images = pickle.load(pkl_file) 
pkl_file.close()

pkl_file = open('model/gesture_label_encoder.pkl', 'rb')
le_gestures = pickle.load(pkl_file) 
pkl_file.close()

print(le_images.inverse_transform([0]))

UDP_IP = "127.0.0.1"
UDP_PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

handTrackingData = WrapperHandTracking()

def main():
    
    image_model = load_model('model/model_images.h5')
    gesture_model = load_model('model/model.h5')

    while True:
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
            xhat = np.array(arr);
            yhat = image_model.predict(xhat.reshape(1, 21, 3))
            image_prediction = le_images.inverse_transform([yhat[0].argmax(axis=0)])[0];
            if image_prediction == 'start_recognizing':
                print('ready for gesture......')
                lines = [];
                count = 0
                while count < 80:
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
                yhat_gesture = gesture_model.predict(input.reshape(1, 80, 21, 3))
                gesture_prediction = le_gestures.inverse_transform([yhat_gesture[0].argmax(axis=0)])[0];
                print(gesture_prediction)
    
    

if __name__ == "__main__":
    main()
# {'turnonlights': array([0., 0., 0., 1.]), 
# 'turnonfan': array([0., 0., 1., 0.]), 
# 'turnofffan': array([1., 0., 0., 0.]), 
# 'turnofflights': array([0., 1., 0., 0.])}