
"""
This script is about uses tensorflow framework to create a 2 layer Hidden N/W to classify different attacks
"""

##########################################
## Author:  Gaurav Vashisth, 23/05/2018 ##
##########################################
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

"""
1. implement dropout ***             done
2. early stopping ****               test
3. use Feature from info gain ** (once I have framework)
4. use different Learning Rates *
5. F1-score   ***
6. AUC, Pr and Recall on valid and test ****          test
7. use validation set                done
"""
import numpy as np
#from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import argparse
from utility import MyMetrics, MMOLogger

_NO_OF_FIRST_UNIT = 100
_NO_OF_SECOND_UNIT = 100
_NO_OF_THIRD_UNIT = 100
_NO_OF_FORTH_UNIT = 100
_NO_OF_FIVE_UNIT = 100

#_KEEP_PROBABILITY = 0.85
_TOTAL_EPOCHS = 1001
_MIN_DELTA = 0.01
_No_OF_CLASSES = 14

_CLASS_WEIGHT = {0: 1.0,
 1: 5.375046163439886,
 2: 2.317441029082675,
 3: 3.719758864168314,
 4: 1.0,
 5: 4.346653545149445,
 6: 4.293963182106703,
 7: 3.9795759746114268,
 8: 10.538561489161944,
 9: 9.36849023651169,
 10: 1.0,
 11: 4.276857783213902,
 12: 5.640721689211033,
 13: 9.902572722441947,
 14: 6.478118478615525 }


#k,dimgray,rosybrown,red,firebrick,maroon,saddlebrown,yellow,lime,blue,indigo,orchid,olive,pink,navy




def kGraph(trainInputData, trainOutputData, validInputData, validOutputData, testInputData, testOutputData, modelDIRNAME,lr=0.001,
           boolSampling=False):



    model = Sequential()
    model.add(Dense(_NO_OF_FIRST_UNIT,activation='tanh',input_dim=trainInputData.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(_NO_OF_SECOND_UNIT, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(_NO_OF_THIRD_UNIT, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(trainOutputData.shape[1], activation='sigmoid'))

    #learning_rate = 0.005
    decay_rate = lr / 100
    #momentum = 0.8

    optimizer = Adam(lr=lr,decay=decay_rate,amsgrad=True)

    filepathCheckpoint = "./Klogs/"+modelDIRNAME+"/checkpoint"
    checkpoint = ModelCheckpoint(filepathCheckpoint, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min')

    tensorBoard = TensorBoard(log_dir='./Klogs/'+modelDIRNAME+'/ADAM', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                                write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)

    metrics = MyMetrics(mylogger, boolBinary=False)

    callbacks_list = [checkpoint,tensorBoard,metrics,earlyStopping]
    #model.load_weights(filepath)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    if boolSampling :
        print("inside correc")
        model.fit(x=trainInputData,y=trainOutputData,validation_data=(validInputData,validOutputData),epochs=_TOTAL_EPOCHS,
              batch_size=100, callbacks=callbacks_list, class_weight=_CLASS_WEIGHT)
    else:
        print("inside non")
        model.fit(x=trainInputData, y=trainOutputData, validation_data=(validInputData, validOutputData),
                  epochs=_TOTAL_EPOCHS,
                  batch_size=100, callbacks=callbacks_list)

    score = model.evaluate(testInputData,testOutputData,batch_size=100,verbose=1)
    y_predict = model.predict(testInputData)

    correct_prediction = np.equal(np.argmax(y_predict, axis=1),np.argmax(testOutputData, axis=1))
    acc = np.mean(correct_prediction/testOutputData.shape[0])

    y_real = np.argmax(testOutputData, axis=1)
    y_hat = np.argmax(y_predict, axis=1)

    mylogger.info(acc)
    mylogger.info(classification_report(y_true=y_real,y_pred=y_hat))
    mylogger.info(score)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingtype", help="Name of training type for creating a folder under Klogs directory")
    parser.add_argument("--isWithoutSampling", type=bool,help="is WithoutSampling training type", default=False)
    parser.add_argument("--logfile",help="logfile name")

    parser.add_argument("--X_train", help="path of X_train")
    parser.add_argument("--y_train", help="path of y_train")

    parser.add_argument("--X_valid", help="path of X_valid")
    parser.add_argument("--y_valid", help="path of y_valid")

    parser.add_argument("--X_test", help="path of X_test")
    parser.add_argument("--y_test", help="path of y_test")

    args = parser.parse_args()

    mylogger = MMOLogger().getLogger(__name__, args.logfile)
    #main(trainingtype=args.trainingtype )
    mylogger.info("training type: %s  is running, check log at %s",args.trainingtype, args.logfile)

    X_train, y_train= np.loadtxt(args.X_train),np.loadtxt(args.y_train)
    X_valid, y_valid = np.loadtxt(args.X_valid),np.loadtxt(args.y_valid)
    X_test, y_test =  np.loadtxt(args.X_test),np.loadtxt(args.y_test)



    mylogger.info("data loaded")
    for learning_rate in [0.002]:
        kGraph(trainInputData=X_train, trainOutputData=y_train, validInputData=X_valid, validOutputData=y_valid,
               testInputData=X_test, testOutputData=y_test, modelDIRNAME=args.trainingtype
               , lr=learning_rate, boolSampling=args.isWithoutSampling )
        mylogger.info('done: ' + str(learning_rate))
