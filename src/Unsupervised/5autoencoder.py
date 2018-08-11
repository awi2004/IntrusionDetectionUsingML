import argparse
import logging
import os

import numpy as np
import keras
from keras import regularizers
from keras.engine.saving import load_model
from keras.optimizers import Adam, SGD

from myCallbacks import MyMetrics
#from keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error

from sklearn.metrics import precision_recall_curve



_NO_EPOCH = 1000
_BATCH_SIZE = 100


_ENCODING_DIM = 2

def getData():

    X_train = np.load("../exprimentData/X_train.npy")
    #y_train = np.load("../dataPreprocessing/exprimentData/y_train.npy")
    X_test = np.load("../exprimentData/X_test.npy")
    #y_test = np.load("../dataPreprocessing/exprimentData/y_test.npy")

    return X_train,X_test #,y_test


def autoEncoder(X_train,X_test, model_name,trainType, first_layer = 18,second_layer = 13,third_layer = 8,four_layer = 3, fifth_layer=2):

    input_dim = X_train.shape[1]


    input_layer = keras.Input(shape=(input_dim, ))

    encoded = keras.layers.Dense(first_layer, activation="relu",
                                 activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoded = keras.layers.Dense(second_layer, activation="relu")(encoded)
    encoded = keras.layers.Dense(third_layer, activation="relu")(encoded)
    encoded = keras.layers.Dense(four_layer, activation="relu")(encoded)
    encoded = keras.layers.Dense(fifth_layer, activation="relu")(encoded)

    # BOTTLE NECK
    block = keras.layers.Dense(_ENCODING_DIM, activation="relu")(encoded)

    decoded = keras.layers.Dense(fifth_layer, activation='relu')(block)
    decoded = keras.layers.Dense(four_layer, activation='relu')(block)
    decoded = keras.layers.Dense(third_layer, activation='relu')(decoded)
    decoded = keras.layers.Dense(second_layer, activation='relu')(decoded)
    decoded = keras.layers.Dense(first_layer, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dim, activation='relu')(decoded)

    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)

    encoder = keras.Model(input_layer, block)
    prd = encoder.predict(X_test)
    np.save("encoded-"+trainType,prd)


    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    filepathCheckpoint = "./Klogs/" + model_name
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepathCheckpoint,
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./encoder/'+trainType,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True ) #,
                              #embeddings_freq=100,  # Store each 100 epochs...
                              #embeddings_layer_names=embeddings_to_monitor,  # this list of embedding layers...
                              #embeddings_metadata=embeddings_metadata)


    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min')
    #metrics = MyMetrics(handler, boolBinary=True)
    csv = keras.callbacks.CSVLogger(trainType+".log")

    autoencoder.fit(X_train, X_train,
                        epochs=_NO_EPOCH,
                        batch_size=_BATCH_SIZE,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        #validation_split=0.2,
                        verbose=1,
                        callbacks=[checkpointer, tensorboard,earlyStopping,csv])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--trainingName", help="Name of training type for creating a folder under Klogs directory")
    parser.add_argument("--model_name",  help="name of teh folder")
    parser.add_argument("--first_layer", type=int,help="no of units in first layer")
    parser.add_argument("--second_layer", type=int,help="no of units in second layer")
    parser.add_argument("--third_layer", type=int,help="no of units in first layer")
    parser.add_argument("--four_layer", type=int, help="no of units in first layer")
    parser.add_argument("--five_layer", type=int, help="no of units in first layer")


    args = parser.parse_args()

    X_train, X_test = getData()
    print("%s started with layers %s  %s %s %s"%(args.model_name,args.first_layer,args.second_layer,args.third_layer,args.four_layer))
    autoEncoder(X_train,X_test, trainType=args.model_name, model_name = args.model_name, first_layer=args.first_layer,
                second_layer=args.second_layer,third_layer=args.third_layer,four_layer= args.four_layer, fifth_layer =args.five_layer)
    #getPrediction(testData=X_test,testlabel=y_test,modelURL='weights.hdf5')

