from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras import optimizers
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.models import Model
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

dosyaismi='deep.h5'
def predict(model1,kontrol):
    if (kontrol==1):
        model1=load_model(dosyaismi)
    y_pred=model1.predict(X_test)
    y_pred = np.around(y_pred)
    y_test_non_category = [ np.argmax(t) for t in y_test ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]
    score, acc = model1.evaluate(X_test, y_test, batch_size=128)
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    f1=f1_score(y_test, y_pred,average='weighted')
    if (kontrol==1):
        print (conf_mat) 
        print(classification_report(y_test, y_pred)) 
        print("ACC=",acc)      
    return acc,f1

def predict_egitim(model1,kontrol):
    if (kontrol==1):
        model1=load_model(dosyaismi)
    y_pred=model1.predict(X_train)
    #y_pred=model1.predict(egitimgiris)
    #print(y_pred)
    y_pred = np.around(y_pred)
    #if (kontrol==1):
    #    print(y_pred)
    #y_test_non_category = [ np.argmax(t) for t in y_test ]
    y_test_non_category = [ np.argmax(t) for t in y_train ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]
    #print(y_pred)
    score, acc = model1.evaluate(X_train, y_train, batch_size=128)
    #score, acc = model1.evaluate(egitimgiris, y_train, batch_size=128)
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    f1=f1_score(y_train, y_pred,average='weighted')
    if (kontrol==1):
        print (conf_mat) 
        print(classification_report(y_train, y_pred)) 
        print("ACC=",acc)  
        
    return acc,f1

kacozellik=13
veriler=pd.read_excel("heart.xlsx")

#veri ön işleme
X= veriler.iloc[:,0:13].values
Y = veriler.iloc[:,13].values

#verilerin egitim ve test icin bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

acc1=0.1
while(acc1<0.62):
    a=2
    baslangic=0
    bitis=6
    for k in range(baslangic,bitis):
        if (k==0):opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        if (k==1):opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        if (k==2):opt = optimizers.SGD(lr=0.01, clipnorm=1.)
        if (k==3):opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        if (k==4):opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        if (k==5):opt = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        #print (egiris,ecikis)
        #model.compile(loss='categorical_crossentropy   mean_absolute_error', optimizer=opt, metrics=['accuracy'])
        model = Sequential()
        aktivasyon="relu"
        model.add(Dense(6, init = 'uniform', activation = aktivasyon, input_dim = 13)) # giriş katmanı
        model.add(Dense(6, init = 'uniform', activation = aktivasyon)) #gizli katmanı
        model.add(Dense(6, init = 'uniform', activation = aktivasyon)) #gizli katmanı
        model.add(Dense(1, init = 'uniform', activation = 'sigmoid')) #çıkış katmanı
        model.compile(optimizer = opt , loss =  'binary_crossentropy' , metrics = ['accuracy'] ) #derleme
        
        for i in range(5):
            #model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=3)
            #acc,fs=predict(model,0)
            acc,fs=predict(model,0)
            #score, acc = model.evaluate(X_test, Y_test, batch_size=128)
            print("i=",i,"k=",k," F score=",fs," ACC=",acc)
            if i==0 and k==baslangic:
                eb=fs
                ebacc=acc
                tutk=k
                model.save("deep.h5")
            if (fs>=eb):
                if (fs==eb):
                    if (acc>ebacc):
                        ebacc=acc
                        tutk=k
                        model.save("deep.h5")
                else:
                    eb=fs#ort mse
                    tutk=k
                    model.save("deep.h5")
    #print (eb,"k=",k)
    print ("En Büyük F Score=",eb, "K değeri=",tutk, "EB ACC=",ebacc)
    print("Test sonuç")
    acc1,fs1=predict(model,1)
    predict_egitim(model,1)
#print("Eğitim sonuç")



