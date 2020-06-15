# gesture_recognition


[![Youtube link](https://img.youtube.com/vi/NarInyCUr_g/0.jpg)](https://www.youtube.com/embed/NarInyCUr_g)

Used the MediaPipe hand tracking for getting the landmark positional information.

Followed the instructions in this [repo](https://github.com/madelinegannon/example-mediapipe-udp) to stream it to a python application using protocol buffers as I cannot code in c++

Once the mediapipe_overrides/mediapipe files are copied to local mediapipe repo, below command will start hand tracking from webcam and stream to udp port

```GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu     --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt```

## Training

The build_training_data.py can be used the build training data files for both gestures and images. These files contain the landmark information from mediapipe.

model_images.h5 is built using train_images.py and trained on hand postures specifically for recognizing the a posture that is used to start the gesture recognition mode. This works similar to the hot word(ex: Hello Siri, Ok Google) in voice services. This is done using a CNN and a Dense layer.

Sample usage for generating image training data

```python build_image_training_data.py --mode=images --name=start_recognizing```

Sample usage for generating gesture training data

```python build_image_training_data.py --mode=gestures --name=turnofflights```

model.h5 is built using train.py and trained on gestures which are sequence of hand postures from mediapipe. This is done using CNN, TimeDistributed and then LSTM with Dropouts 

```python train.py```

## Prediction Loop

predict_image.py is the script that runs in a loop trying to detect the hand pose(using model_images.h5) to start the gesture recognition(using model.h5) by collecting a sequence of 80 poses after the start.

```python predict_image.py```

## Next Steps

Convert this to a tflite model and try to use directly in mediapipe and use from raspberry pi.


Thanks to 
* Google mediapipe for hand detection and tracking - https://github.com/google/mediapipe
* This Github repo for showing how to stream information from media pipe to external applications using UDP -  https://github.com/madelinegannon/example-mediapipe-udp
