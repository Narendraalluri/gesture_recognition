import os
import matplotlib.pyplot as plt

# Example
input_shape = (40, 21, 3)


def process_file(file):
    print(file)
    f = open(file, "r")
    lines = f.readlines()
    for line in lines:
        plt.clf()
        arr = [[float(x) for x in word.split(' ')] for word in line.split(',')]
        x, y, z = zip(*arr)
        plt.plot(x, y);
        plt.pause(.0001)

def process_folder(subfolder):
    return [ process_file(f) for f in os.scandir(subfolder) ]

def main():
    process_folder('/Users/narendra/Work/gesture_recognition/data/gestures/off')


if __name__ == "__main__":
    main()