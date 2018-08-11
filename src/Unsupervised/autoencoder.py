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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
handler = logging.FileHandler('autoEncoder.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

_NO_EPOCH = 1000
_BATCH_SIZE = 100


_ENCODING_DIM = 2

def getData():

    X_train = np.load("../exprimentData/X_train.npy")
    #y_train = np.load("../dataPreprocessing/exprimentData/y_train.npy")
    X_test = np.load("../exprimentData/X_test.npy")
    #y_test = np.load("../dataPreprocessing/exprimentData/y_test.npy")

    return X_train,X_test #,y_test

#to train autenocder
def autoEncoder(X_train,X_test, model_name,trainType, first_layer = 18,second_layer = 13,third_layer = 8,four_layer = 3):

    input_dim = X_train.shape[1]


    input_layer = keras.Input(shape=(input_dim, ))

    encoded = keras.layers.Dense(first_layer, activation="relu",
                                 activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoded = keras.layers.Dense(second_layer, activation="relu")(encoded)
    encoded = keras.layers.Dense(third_layer, activation="relu")(encoded)
    encoded = keras.layers.Dense(four_layer, activation="relu")(encoded)
    #encoded = keras.layers.Dense(_FIFTH_LAYER, activation="relu")(encoded)

    # BOTTLE NECK
    block = keras.layers.Dense(_ENCODING_DIM, activation="relu")(encoded)

    #decoded = keras.layers.Dense(_FOURTH_LAYER, activation='relu')(block)
    decoded = keras.layers.Dense(four_layer, activation='relu')(block)
    decoded = keras.layers.Dense(third_layer, activation='relu')(decoded)
    decoded = keras.layers.Dense(second_layer, activation='relu')(decoded)
    decoded = keras.layers.Dense(first_layer, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dim, activation='relu')(decoded)

    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)

    encoder = keras.Model(input_layer, block)
    prd = encoder.predict(X_test)
    np.save("encoded-"+trainType,prd)


    autoencoder.compile(optimizer='rmsprop',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    filepathCheckpoint = "./Klogs/" + model_name + "/checkpoint"
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=filepathCheckpoint,
                                   verbose=0,
                                   save_best_only=True)

    """embeddings_to_monitor = ['embeddings_{}'.format(i)
                             for i in range(4)]

    metadata_file_name = 'metadata.tsv'
    embeddings_metadata = {layer_name: metadata_file_name
                           for layer_name in embeddings_to_monitor}"""

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



def getPrediction(testData,testlabel,modelURL):
    model = load_model(modelURL)
    prediction = model.predict(testData)
    mse =np.mean(np.power(X_test - prediction, 2), axis=1)
    #error_df = pd.DataFrame({'reconstruction_error': mse,
                           #  'true_class': testlabel})
    #precision, recall, th = precision_recall_curve(error_df.true_class, keras error_df.reconstruction_error)
    #f1 = (2*precision*recall)/(precision+recall)

    #max_pr = precision(np.argmax(f1))
    #max_rc = recall(np.argmax(f1))
    print(mse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--trainingName", help="Name of training type for creating a folder under Klogs directory")
    parser.add_argument("--model_name",  help="name of teh folder")
    parser.add_argument("--first_layer", type=int,help="name of teh folder")
    parser.add_argument("--second_layer", type=int,help="name of teh folder")
    parser.add_argument("--third_layer", type=int,help="name of teh folder")
    parser.add_argument("--four_layer", type=int, help="name of teh folder")

    args = parser.parse_args()

    X_train, X_test = getData()
    print("%s started with layers %s  %s %s %s"%(args.model_name,args.first_layer,args.second_layer,args.third_layer,args.four_layer))
    autoEncoder(X_train,X_test, trainType=args.model_name, model_name = args.model_name, first_layer=args.first_layer,
                second_layer=args.second_layer,third_layer=args.third_layer,four_layer= args.four_layer)
    #getPrediction(testData=X_test,testlabel=y_test,modelURL='weights.hdf5')

