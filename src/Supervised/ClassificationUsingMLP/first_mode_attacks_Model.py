
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

_CLASS_WEIGHT_ATTACKS = {0: 3.582700529727665,
 1: 1.0,
 2: 1.927237498908217,
 3: 1.0,
 4: 2.554135293926962,
 5: 2.5015335269942036,
 6: 2.1870401970917657,
 7: 8.768561557636406,
 8: 7.582937891978666,
 9: 1.0,
 10: 2.4842578044342725,
 11: 3.848580631808281,
 12: 8.121934392711353,
 13: 4.686412268508123}



#k,dimgray,rosybrown,red,firebrick,maroon,saddlebrown,yellow,lime,blue,indigo,orchid,olive,pink,navy



"""def getData():

    X_train = np.loadtxt("./CompleteWithoutSampling/X_train.np")
    y_train = np.loadtxt("./CompleteWithoutSampling/y_train.np")

    X_test = np.loadtxt("./CompleteWithoutSampling/X_test.np")
    y_test = np.loadtxt("./CompleteWithoutSampling/y_test.np")

    X_valid = np.loadtxt("./CompleteWithoutSampling/X_valid.np")
    y_valid = np.loadtxt("./CompleteWithoutSampling/y_valid.np")
    mylogger.info("data loaded")
    return X_train,y_train,X_test,y_test,X_valid,y_valid


"""




def kGraph(trainInputData, trainOutputData, validInputData, validOutputData, testInputData, testOutputData, modelDIRNAME,lr=0.001,
           boolWithoutSampling=False):



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

    metrics = MyMetrics(mylogger, boolBinary=boolWithoutSampling)

    callbacks_list = [checkpoint,tensorBoard,metrics,earlyStopping]
    #model.load_weights(filepath)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    if not boolWithoutSampling :

        model.fit(x=trainInputData,y=trainOutputData,validation_data=(validInputData,validOutputData),epochs=_TOTAL_EPOCHS,
              batch_size=100, callbacks=callbacks_list, class_weight=_CLASS_WEIGHT_ATTACKS)
    else:
        print("inside sampling")
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
               , lr=learning_rate, boolWithoutSampling=args.isWithoutSampling )
        mylogger.info('done: ' + str(learning_rate))
