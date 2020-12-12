import pandas as pd
import numpy
def comE(y_true,y_pred):
    from sklearn.metrics import confusion_matrix
    #####------------   tn，fp, fn, tp  ------------###
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
    SN = TP*1.0/(TP + FN)*1.0 
    #specificity, true negative rate:TNR = TN / (TN + FP)
    SP = TN / (TN + FP)*1.0  
    #precision, prositive predictive value:PPV = TP / (TP + FP)
    precision = TP / (TP + FP)*1.0
    #negative predictive value:NPV = TN / (TN + FN)
    NPV = TN / (TN + FN)*1.0
    # F1 score is the harmonic mean of precision and sensitivity
    F1= 2*TP / (2*TP + FP+FN)*1.0
    
    return SN, SP,precision, NPV,F1


PRE_PSPPone_9= open('./result/PRE_CNN_9_27_PSPP_predict_probablity_earlystopping_10_fold_corss_validation.txt','w')



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 



def my_model():
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
    
    K.clear_session()
    
    channels = 1
    nb_classes = 2
    batch_size = 16
    epochs = 100
    filters = 32  
    kernel_size = 5
    pooling_size = 2
    img_rows = 9
    img_columns = 27
    color_type = 1
    input1=Input(shape=(9,27,1))
    # reshape=Reshape((9,27,1))(input)

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
    return model 




data = pd.read_excel('PRE_PSPP_9.xlsx')
y = data['Label']
X = data.loc[:,'pssm0':'psa26']
print("load finish ！")


from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np

cv = StratifiedKFold(n_splits=10) 



AUC=[]
MCC=[]
ACC=[]
SN =[]
SP=[]
precision=[]
NPV=[]
F1=[]

from numpy import array
import numpy as np
from keras.utils import to_categorical

k=1  
for train, test in cv.split(X, y):

    train_Features = X.iloc[train]
    train_Label = y[train]
    test_Features = X.iloc[test]
    test_Label = y[test]

    print(train_Features.shape, train_Label.shape)
    print(test_Features.shape, test_Label.shape)
    train_len = len(train_Features)
    test_len = len(test_Features)

    train_Features = array(train_Features).reshape(train_len,9,27,1)
    print(train_Features.shape,type(train_Features))
    train_Label = array(train_Label).reshape(train_len)
    print(train_Label.shape,type(train_Label))

    test_Features = array(test_Features).reshape(test_len,9,27,1)
    print(test_Features.shape,type(test_Features))
    test_Label = array(test_Label).reshape(test_len)
    print(test_Label.shape,type(test_Label))

    X_train = train_Features
    y_train = train_Label
    X_test = test_Features
    y_test = test_Label

    from keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_test= to_categorical(y_test)
    
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2,verbose=1, mode='auto')
    
    
    ## 加载model
    model = my_model()
    
    ##--------------------------------------##
#     batch_size = 16
#     epochs = 100
    ##---------------------------------------##
    
    hist = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_split = 0.2,callbacks=[early_stopping])   
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    y_true = np.argmax(y_test,axis=1)
    print("\tmatthews_corrcoef: %1.3f" % metrics.matthews_corrcoef(y_true, y_pred))
    MCCv=metrics.matthews_corrcoef(y_true, y_pred)
    MCC.append(MCCv)#
    print("\taccuracy_score: %1.3f\n" % metrics.accuracy_score(y_true, y_pred))
    ACCv=metrics.accuracy_score(y_true, y_pred)
    ACC.append(ACCv)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    AUCv = metrics.auc(fpr, tpr)
    print('AUC: %.3f\n' % AUCv)
    AUC.append(AUCv)

    SNv,SPv,precisionv,NPVv,F1v = comE(y_true, y_pred)## 其他评价指标
    ## y_true, y_pred 
    SN.append(SNv)
    SP.append(SPv)
    precision.append(precisionv)
    NPV.append(NPVv)
    F1.append(F1v)
    ####------------ write file---------##
    PRE_PSPPone_9.write("第"+str(k)+"折交叉验证"+'\t')
    PRE_PSPPone_9.write("\tAUC: \t"+str(AUCv))
    PRE_PSPPone_9.write("\tMCC: \t"+str(MCCv))
    PRE_PSPPone_9.write("\tACC: \t"+str(ACCv))
    PRE_PSPPone_9.write("\tSN: \t"+str(SNv))
    PRE_PSPPone_9.write("\tSP: \t"+str(SPv))
    PRE_PSPPone_9.write("\tprecision: \t"+str(precisionv))
    PRE_PSPPone_9.write("\tNPV: \t"+str(NPVv))
    PRE_PSPPone_9.write("\tF1: \t"+str(F1v))
    PRE_PSPPone_9.write('\n')

    k=k+1  

print("AUC mean: %.3f +/- %.3f :"% (np.mean(AUC),np.std(AUC)))
print("MCC mean: %.3f +/- %.3f :"% (np.mean(MCC),np.std(MCC)))
print("ACC mean: %.3f +/- %.3f :"% (np.mean(ACC),np.std(ACC)))
print("SN mean: %.3f +/- %.3f :"% (np.mean(SN),np.std(SN)))
print("SP mean: %.3f +/- %.3f :"% (np.mean(SP),np.std(SP)))
print("precision mean: %.3f +/- %.3f :"% (np.mean(precision),np.std(precision)))
print("NPV mean: %.3f +/- %.3f :"% (np.mean(NPV),np.std(NPV)))
print("F1 mean: %.3f +/- %.3f :"% (np.mean(F1),np.std(F1)))

####---------------write file  -------------##
PRE_PSPPone_9.write("\tAUCmean: \t"+str(np.mean(AUC)))
PRE_PSPPone_9.write("\tMCCmean: \t"+str(np.mean(MCC)))
PRE_PSPPone_9.write("\tACCmean: \t"+str(np.mean(ACC)))
PRE_PSPPone_9.write("\tSNmean: \t"+str(np.mean(SN)))
PRE_PSPPone_9.write("\tSPvmean: \t"+str(np.mean(SP)))
PRE_PSPPone_9.write("\tprecisionmean: \t"+str(np.mean(precision)))
PRE_PSPPone_9.write("\tNPVmean: \t"+str(np.mean(NPV)))
PRE_PSPPone_9.write("\tF1mean: \t"+str(np.mean(F1)))
PRE_PSPPone_9.write('\n\n\n')
PRE_PSPPone_9.close()