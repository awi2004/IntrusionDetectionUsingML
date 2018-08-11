import logging

from keras import Sequential
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, recall_score,precision_score, average_precision_score

from sklearn.manifold import TSNE
class MyMetrics(Callback):


    def __init__(self,mylogger,boolBinary=False):
        self.logger = mylogger

        self.boolBinary = boolBinary
        #self._val_f1 = 0.0
        #self._val_recall = 0.0
        #self._val_precision = 0.0
        #self.avg_precision = 0.0



    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        #self.auc_roc = []
        #self.average_precision = []


    def on_epoch_end(self, epoch, logs={}):


        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # self.auc_roc.append(_auc)
        precision = dict()
        recall = dict()
        average_precision = dict()

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(val_targ.ravel(),
                                                                        val_predict.ravel())
        #average_precision["micro"] = average_precision_score(val_targ, val_predict,
        #                                                     average="micro")
        _avg_precision = average_precision_score(val_targ, val_predict,
                                                           average="micro")
        if (self.boolBinary):
            _auc = roc_auc_score(val_targ, val_predict, average='weighted')  # in case of  of binary
            self.logger.info('AUC score: %f — val_f1: %f — val_precision: %f — val_recall: %f' % (_auc, _val_f1, _val_precision, _val_recall))
        else:
            self.logger.info('Average precision score: %f — val_f1: %f — val_precision: %f — val_recall: %f' % (
                _avg_precision, _val_f1, _val_precision, _val_recall))
        return





class MMOLogger():
    def getLogger(self,loggerName,loggerURL):
        #self.loggerConfig = loggerConfig
        self.loggerName = loggerName
        self.loggerURL = loggerURL

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(self.loggerName)
        hdlr = logging.FileHandler(self.loggerURL)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

        return logger



def get_model_visualiation(X_sample, true_label, modelpath):
    model = Sequential()
    model.load_weights(modelpath)

    y_predict = model.predict_classes(X_sample)
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X_sample)

