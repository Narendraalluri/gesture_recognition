#!/usr/bin/python
import socket
import argparse
from protocol_buffer_model.wrapper_hand_tracker_pb2 import WrapperHandTracking
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import uuid
from pathlib import Path
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

handTrackingData = WrapperHandTracking()

def main(mode, name):
    
    os.makedirs('data/{0}/{1}'.format(mode, name), exist_ok=True)
    
    count = 40

    while True:
        plt.clf()

        if count == 40:
            file_name = uuid.uuid4()
            file_path = 'data/{0}/{1}/{2}.txt'.format(mode, name, file_name)
            Path(file_path).touch()
            lines = [];
            first = True
            count = 0
            print('Recording')


        data, addr = sock.recvfrom(1024)
        handTrackingData.ParseFromString(data)

        file_mode = 'a' if mode == 'gestures' else 'r+'

        with open(file_path, file_mode) as the_file:
            line = ""
            if len(handTrackingData.landmarks.landmark) == 21 and handTrackingData.landmarks.landmark[0].x > 0.0:
                arr = []
                for i in range(len(handTrackingData.landmarks.landmark)):
                    arr.append([])
                    landmark = handTrackingData.landmarks.landmark[i]
                    arr[i].append(landmark.x)
                    arr[i].append(landmark.y)
                    arr[i].append(landmark.z)
                    line += "{0} {1} {2},".format(landmark.x, landmark.y, landmark.z)
                    
                the_file.write(line[:-1]) if first else the_file.write("\n" + line[:-1])
                first = False
                x, y, z = zip(*arr)
                count = count + 1
                plt.plot(x, y);
                plt.pause(.001)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='images or gestures', required=True)
    parser.add_argument('--name', help='name of the pose', required=True)
    args = parser.parse_args()
    main(args.mode, args.name)