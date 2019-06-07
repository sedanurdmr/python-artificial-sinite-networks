import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_excel('heart.xlsx') #veriler yüklenir.

#veri ön işleme
X= veriler.iloc[:,0:13].values
Y = veriler.iloc[:,13].values

#verilerin egitim ve test icin işlemleri için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin ölçeklenmesi : amaç verileri 0 1 arasına sıkıştırmak
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#yapay sinir ağı
import keras
from keras.models import Sequential #kerasa bir yapay sinir ağı oluştur demek istiyoruz
from keras.layers import Dense #katman oluşturmak için gerekli nesne ,nöronları tanımlamak için

classifier = Sequential() #yapay sinir ağı oluşmuş oldu , objenin ismi
#6 1.gizli katmandaki nöron sayısı , 13 giriş katmanındaki nöron sayısı
#neural networku initialize(başlatmak) için init kullanıyoruz ve 0'a yakın olmasını istiyoruz.uniform dağılım ile yapay sinir ağımızdaki sinapsislerin üzerine verileri ilkle diyoruz.
#activation fonksiyonunda istediğimiz fonksiyonu kullanabiliriz.
#13 yapay sinir ağının girişinde kaç veri olduğudur.
classifier.add(Dense(6, init = 'uniform', activation = 'elu' , input_dim = 13)) #giriş katman

classifier.add(Dense(6, init = 'uniform', activation = 'elu')) #gizli katman

classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid')) #çıkış katman #sigmoid logaritmik bir fonksiyondur.
#çıkış verilerimiz 1 ve 0 lardan oluştuğu için loss fonksiyonlarından binar_crossentropy yi seçtik.
classifier.compile(optimizer = 'adadelta', loss =  'binary_crossentropy' , metrics = ['accuracy'] ) #derleme işlemi
#50 epochs ile neural networkumuzu eğitecek.
classifier.fit(X_train, y_train, epochs=50)



from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
y_pred = clf.predict(X_test)

y_pred = (y_pred > 0.5) #y_pred 0.5 in altında ise false, değilse true döndürecek.dolayısıyle trueler 1, falseler 0 olacaktır.

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #gerçek değerle ile tahmin değeler alınacak confusion matrix ekrana yazılmıştır.

print(cm)