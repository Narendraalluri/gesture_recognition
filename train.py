import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, InputLayer, Dropout
import random
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model
import json
import h5py
import pickle

def fix_layer0(filename, batch_input_shape, dtype):
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

# Example
input_shape = (80, 21, 3)

def build_model(label):
    model = Sequential()
    # define CNN model
    model.add(InputLayer(input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(label, activation='softmax'))

   
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.build()
    return model

def process_file(label, file):
    padded = np.zeros(input_shape)
    f = open(file.path, "r")
    lines = f.readlines()
    result_list = [[word.split(' ') for word in line.split(',')] for line in lines]
    result = np.array(result_list)
    print(result)
    print(file, result.shape, padded.shape)
    min1st = min(result.shape[0],padded.shape[0])
    min2nd = min(result.shape[1],padded.shape[1])
    min3rd = min(result.shape[2],padded.shape[2])
    padded[:min1st ,:min2nd , :min3rd] = result[:min1st ,:min2nd , :min3rd]
    return (padded, label)
    

def process_folder(subfolder):
    return [ process_file(subfolder.name, f) for f in os.scandir(subfolder) ]

def main():
    labels = [ f.name for f in os.scandir('data/gestures') if f.is_dir() ]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    label_mapping = {}
    for i in range(len(labels)):
        label_mapping[labels[i]] = onehot_encoded[i]

    output = open('model/gesture_label_encoder.pkl', 'wb')
    pickle.dump(label_encoder, output)
    output.close()

    result = [ process_folder(f) for f in os.scandir('data/gestures') if f.is_dir() ]
    flat_list = [item for sublist in result for item in sublist]

    random.shuffle(flat_list)

    model = build_model(len(labels))

    flat_list = [ (x, label_mapping[y]) for (x, y) in flat_list ]
    train, test = train_test_split(flat_list, test_size=0.2)
    x_train = np.array([x for (x, y) in train])
    y_train = np.array([y for (x, y) in train])
    x_test = np.array([x for (x, y) in test])
    y_test = np.array([y for (x, y) in test])
    model.build()
    history=model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_test,y_test))
    score, acc = model.evaluate(x_test,y_test,batch_size=1,verbose=0)
    print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
    model.build()
    model.save('model/model.h5')
    fix_layer0('model/model.h5', [None, 80, 21, 3], 'float32')
    print(label_mapping)


if __name__ == "__main__":
    main()