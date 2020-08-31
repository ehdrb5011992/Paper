import tensorflow as tf
from tensorflow.keras import backend as K


def smoothed_categorical_crossentropy(y_true,y_pred,smoothing_param=0.1,classes=7) :
    if smoothing_param >0:
        smooth_positives = 1.0-smoothing_param
        smooth_negatives = smoothing_param / classes
        y_true = y_true * smooth_positives + smooth_negatives

        return K.categorical_crossentropy(y_true,y_pred)
    