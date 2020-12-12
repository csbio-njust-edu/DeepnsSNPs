
PRE_PSPP_9 = open('./result/PRE_CNN_9_27_PSSM.txt','w')

#import pandas as pd
#data = pd.read_excel('PRE_PSPP_9.xlsx')
#y = data['Label']
#X = data.loc[:,'pssm0': 'psa26']

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
#PRE_PSPP_9_train = pd.DataFrame()
#X_train.insert(0,'Label',y_train)
#PRE_PSPP_9_train=X_train
#PRE_PSPP_9_train.to_excel('./result/PRE_PSPP_9_train.xlsx',index = False)

#PRE_PSPP_9_test = pd.DataFrame()
#X_test.insert(0,'Label',y_test)
#PRE_PSPP_9_test=X_test
#PRE_PSPP_9_test.to_excel('./result/PRE_PSPP_9_test.xlsx',index = False)

## comE---##
import pandas as pd
def comE(y_true,y_pred):
    from sklearn.metrics import confusion_matrix
    #####------------   定义：tn，fp, fn, tp  ------------###
    def tn(y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)[1, 1]

    TN = tn(y_true, y_pred)
    FP = fp(y_true, y_pred)
    TP = tp(y_true, y_pred)
    FN = fn(y_true, y_pred)

    #sensitivity, recall, hit rate, true positive rate ：TPR = TP / (TP + FN)
    SN = TP*1.0/(TP + FN)*1.0 ## 
    #specificity, true negative rate:TNR = TN / (TN + FP)
    SP = TN / (TN + FP)*1.0  ## 
    #precision, prositive predictive value:PPV = TP / (TP + FP)
    precision = TP / (TP + FP)*1.0
    #negative predictive value:NPV = TN / (TN + FN)
    NPV = TN / (TN + FN)*1.0
    # F1 score is the harmonic mean of precision and sensitivity
    F1= 2*TP / (2*TP + FP+FN)*1.0

    return SN, SP,precision, NPV,F1


##############################-------------------PSSM ------------------------------------######

import pandas as pd
data = pd.read_excel('./result/PRE_PSPP_9_train.xlsx')
y = data['Label']
X = data.loc[:,'pssm0':'pssm179']
print(y.shape,X.shape)

data = pd.read_excel('./result/PRE_PSPP_9_test.xlsx')
yy = data['Label']
XX = data.loc[:,'pssm0':'pssm179']
print(yy.shape,XX.shape)


import pandas as pd
import numpy

train_Features = X
train_Label = y
test_Features = XX
test_Label = yy

print(train_Features.shape, train_Label.shape)
print(test_Features.shape, test_Label.shape)


train_len = len(train_Features)
test_len = len(test_Features)

from numpy import array
train_Features = array(train_Features).reshape(train_len,9,20,1)
print(train_Features.shape,type(train_Features))
train_Label = array(train_Label).reshape(train_len)
print(train_Label.shape,type(train_Label))

test_Features = array(test_Features).reshape(test_len,9,20,1)
print(test_Features.shape,type(test_Features))
test_Label = array(test_Label).reshape(test_len)
print(test_Label.shape,type(test_Label))



import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.optimizers import SGD,Adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D,Concatenate
from keras.layers import ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation,Input
from keras.layers import Add,GlobalMaxPooling2D
from keras.layers import  LSTM,ConvLSTM2D,concatenate,GlobalMaxPooling2D
from  keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.layers.core import Lambda


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

#Parameters
channels = 1
nb_classes = 2
batch_size = 16
epochs = 100
filters = 32
kernel_size = 5
pooling_size = 2
img_rows = 9
img_columns = 20
color_type = 1
input1=Input(shape=(9,20,1))
# reshape=Reshape((9,20,1))(input)

## Scale_layer 1
conv11 = Convolution2D(128,(3, 3), strides=(1,1), activation='relu',padding='same')(input1)
BNor11 = BatchNormalization()(conv11)
MaxPool11 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor11)

conv12 = Convolution2D(128,(5, 5), strides=(1,1), activation='relu',padding='same')(input1)
BNor12 = BatchNormalization()(conv12)
MaxPool12 = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor12)

conv13 = Convolution2D(128,(7, 7), strides=(1,1), activation='relu',padding='same')(input1)
BNor13 = BatchNormalization()(conv13)
MaxPool13 = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor13)

add1 =Add()([MaxPool11,MaxPool12,MaxPool13])
BN1 = BatchNormalization()(add1)


## Scale_layer 2
conv21 = Convolution2D(256,(3, 3), strides=(1,1), activation='relu',padding='same')(BN1 )
BNor21 = BatchNormalization()(conv21)
MaxPool21 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor21)

conv22 = Convolution2D(256,(5, 5), strides=(1,1), activation='relu',padding='same')(BN1 )
BNor22 = BatchNormalization()(conv22)
MaxPool22 = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor22)

conv23 = Convolution2D(256,(7, 7), strides=(1,1), activation='relu',padding='same')(BN1 )
BNor23 = BatchNormalization()(conv23)
MaxPool23 = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor23)

add2 =Add()([MaxPool21,MaxPool22,MaxPool23])
BN2 = BatchNormalization()(add2)

## Scale_layer 3
conv31 = Convolution2D(512,(3, 3), strides=(1,1), activation='relu',padding='same')(BN2)
BNor31 = BatchNormalization()(conv31)
MaxPool31 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor31)

conv32 = Convolution2D(512,(5, 5), strides=(1,1), activation='relu',padding='same')(BN2)
BNor32 = BatchNormalization()(conv32)
MaxPool32 = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor32)

conv33 = Convolution2D(512,(7, 7), strides=(1,1), activation='relu',padding='same')(BN2)
BNor33 = BatchNormalization()(conv33)
MaxPool33 = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor33)

add3 =Add()([MaxPool31,MaxPool32,MaxPool33])
BN3 = BatchNormalization()(add3)

# g = GlobalMaxPooling2D()(BN3)

 # Add Fully Connected Layer
flat = Flatten()(BN3)
den1 = Dense(512, activation='relu')(flat )
drop1 = Dropout(0.4)(den1)
den2 = Dense(256, activation='relu')(drop1)
drop2 = Dropout(0.4)(den2)
pred = Dense(nb_classes, activation='softmax')(drop2)


model=Model(input = input1,outputs = pred)
model.summary()
SGD = SGD(lr=0.0001)
model.compile(loss='categorical_crossentropy',
            optimizer=SGD,
            metrics=['accuracy'])


#TRAINING
X_train = train_Features
y_train = train_Label
X_test = test_Features
y_test = test_Label

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

#---------------------------------------------------early stopping -----------------------##
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2,verbose=1, mode='auto')

hist = model.fit(X_train, y_train, batch_size=batch_size,nb_epoch=epochs,
               verbose=1,validation_split = 0.2,callbacks=[early_stopping])
##-------------------------------------------------early stopping end -------------------##
#EVALUATION
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

PRE_PSPP_9.write('\n\n\n---------- PSSM ----------------------\n\n\n')
PRE_PSPP_9.write('\n---------- CNN_9_20 ----------------------\n')
PRE_PSPP_9.write('\t Test loss:\t'+ str(score[0]))
PRE_PSPP_9.write('\n Test accuracy:\t'+ str(score[1]))


# Compute confusion matrix
Y_pred = model.predict(X_test,batch_size=batch_size, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

###----------------------------  3. AUC  ----------------------------------##
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(np.argmax(y_test,axis=1), y_pred)
roc_auc = metrics.auc(fpr, tpr)
print('AUC: %.3f\n' % roc_auc)
PRE_PSPP_9.write('\n AUC::\t'+ str(roc_auc))
##-----------------------  4.other evaluation values  ---------------##
y_test = np.argmax(y_test,axis=1)
sensitivity, specificity,precision, NPV,F1 = comE(y_test,y_pred)
print("\tmatthews_corrcoef: %1.3f\n" % metrics.matthews_corrcoef(y_test, y_pred))
print("\tcohen_kappa_score: %1.3f" % metrics.cohen_kappa_score(y_test, y_pred))
print("\taccuracy_score: %1.3f" % metrics.accuracy_score(y_test, y_pred))
sensitivity, specificity,precision, NPV,F1 = comE(y_test,y_pred)
print("\tsensitivity: %1.3f" % sensitivity)
print("\tspecificity: %1.3f" % specificity)
print("\tprecision: %1.3f" % precision)
print("\tnegative predictive value: %1.3f" % NPV)
print("\tF1: %1.3f" % F1)
PRE_PSPP_9.write('\nmatthews_corrcoef:\t'+ str(metrics.matthews_corrcoef(y_test, y_pred)))
PRE_PSPP_9.write('\ncohen_kappa_score:\t'+ str(metrics.cohen_kappa_score(y_test, y_pred)))
PRE_PSPP_9.write('\naccuracy_score:\t'+ str(metrics.accuracy_score(y_test, y_pred)))
PRE_PSPP_9.write('\nsensitivity: :\t'+ str(sensitivity))
PRE_PSPP_9.write('\nspecificity:\t'+ str(specificity))
PRE_PSPP_9.write('\nprecision:\t'+ str(precision))
PRE_PSPP_9.write('\nnegative predictive value-NPV:\t'+ str(NPV))
PRE_PSPP_9.write('\nF1:\t'+ str(F1))


import numpy as np
def predict_prob_com(Y_pred, alpha):
    len_Y_pred = len(Y_pred)
    y_pred = np.ones(len_Y_pred)
    for i in range(len_Y_pred):
        if(Y_pred[i]>= alpha ):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred


# Compute evaplution values
from sklearn import metrics
import numpy as np
Y_pred = model.predict(X_test,batch_size=batch_size, verbose=1)
##[[0.8,0.8],[0.1,0.9]]
p = np.array(Y_pred)
i = 1 # 
temp = p[:, i] ## 
Y_pred = temp.tolist() ##

alpha_list = [0.1, 0.15, 0.17,0.2,0.25,0.28, 0.3,0.35,0.37, 0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,
              0.48,0.49, 0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,
              0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80,0.81,
              0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97]
              
PRE_PSPP_9.write('\n PRE_CNN_9_20_predict_proba\n')
for alpha in alpha_list:
    y_pred = predict_prob_com(Y_pred,alpha)
    print("--------------- alpha =    ",alpha)
    PRE_PSPP_9.write('\n alpha::\t'+ str(alpha))

    ##---- AUC ---##
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC: %.3f\n' % roc_auc)
    PRE_PSPP_9.write('\n AUC::\t'+ str(roc_auc))

    #---- MCC, ACC, and others---##
    sensitivity, specificity,precision, NPV,F1 = comE(y_test,y_pred)
    print("\tmatthews_corrcoef: %1.3f\n" % metrics.matthews_corrcoef(y_test, y_pred))
    print("\tcohen_kappa_score: %1.3f" % metrics.cohen_kappa_score(y_test, y_pred))
    print("\taccuracy_score: %1.3f" % metrics.accuracy_score(y_test, y_pred))
    sensitivity, specificity,precision, NPV,F1 = comE(y_test,y_pred)
    print("\tsensitivity: %1.3f" % sensitivity)
    print("\tspecificity: %1.3f" % specificity)
    print("\tprecision: %1.3f" % precision)
    print("\tnegative predictive value: %1.3f" % NPV)
    print("\tF1: %1.3f\t" % F1)


    PRE_PSPP_9.write('\nmatthews_corrcoef:\t'+ str(metrics.matthews_corrcoef(y_test, y_pred)))
    PRE_PSPP_9.write('\ncohen_kappa_score:\t'+ str(metrics.cohen_kappa_score(y_test, y_pred)))
    PRE_PSPP_9.write('\naccuracy_score:\t'+ str(metrics.accuracy_score(y_test, y_pred)))
    PRE_PSPP_9.write('\nsensitivity: :\t'+ str(sensitivity))
    PRE_PSPP_9.write('\nspecificity:\t'+ str(specificity))
    PRE_PSPP_9.write('\nprecision:\t'+ str(precision))
    PRE_PSPP_9.write('\nnegative predictive value-NPV:\t'+ str(NPV))
    PRE_PSPP_9.write('\nF1:\t'+ str(F1))




PRE_PSPP_9.close()

#SAVE WHOLE MODEL (architecture + weights + training configuration[loss,optimizer] +
# state of the optimizer). This allows to resume training where we left off.
#model.save("./result/PRE_CNN_9_27_early_stopping.h5")
del model

#------------------------------------------------ end -----------------------------##

