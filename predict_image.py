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

    devices = ['fan', 'lights']
    actions = ['on', 'off']

    current_device = None
    current_action = None

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
                print(yhat[0])
                gesture_recognized = False
                while gesture_recognized != True:
                    lines = [];
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
                    yhat_gesture = gesture_model.predict(input.reshape(1, 40, 21, 3))
                    gesture_prediction = le_gestures.inverse_transform([yhat_gesture[0].argmax(axis=0)])[0];

                    print(gesture_prediction)
                    if gesture_prediction in devices:
                        current_device = gesture_prediction
                    if gesture_prediction in actions:
                        current_action = gesture_prediction
                    
                    if current_action is not None and current_device is not None:
                        print('Turning {0} the {1}'.format(current_action,current_device))
                        current_device = None
                        current_action = None
                        gesture_recognized = True
    
    

if __name__ == "__main__":
    main()
