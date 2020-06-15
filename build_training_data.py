#!/usr/bin/python
import socket
from wrapper_hand_tracker_pb2 import WrapperHandTracking
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import uuid
from pathlib import Path

UDP_IP = "127.0.0.1"
UDP_PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

handTrackingData = WrapperHandTracking()

def main():
    folder_name = sys.argv[1]
    file_name = uuid.uuid4()
    os.makedirs('data/gestures/{0}'.format(folder_name), exist_ok=True)
    file_path = 'data/gestures/{0}/{1}.txt'.format(folder_name, file_name)

    lines = [];
    first = True
    while True:
        plt.clf()
        data, addr = sock.recvfrom(1024)
        handTrackingData.ParseFromString(data)
        with open(file_path, 'a') as the_file:
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
                plt.plot(x, y);
                plt.pause(.001)
    
if __name__ == "__main__":
    main()