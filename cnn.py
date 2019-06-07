from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator #resim okumaya yarayan bir kütüphanedir.Resimleri tek tek okuyarak öznitelik çıkarımı yapmaktadır.

train_datagen = ImageDataGenerator(rescale = 1./255, #resim işlemeye yarayan filtrelerdir.
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('veriseti/egitim',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('veriseti/Test',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')


classifier = Sequential()

#2 boyutlu bir Convolution neural network tanımlanmıştır.
#Renkli resimleride RGB şeklinde 3 katman vardır ve bunlar 3 ayrı maris şeklinde incelenir.
#64,64 okunan resimler 64,64 boyutuna indirilir.
#2 convulation(evrişim) ve 2 pooling(havuzlama) adımlarından geçmiştir.
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Pooling-->havuzlama
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#convolution katmanı-->evrişim
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening-->düzleştirmek
classifier.add(Flatten())

#YSA
classifier.add(Dense(output_dim = 128, activation = 'relu'))#128 bitlik output verilmiştir.
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#CNN
classifier.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])

#CNN ve resimler
#yapay sinir ağının eğitilmesi
#training_set verilerinden epoch sayısını ve her epoch sırasında kaç veri okyacağını belirliyoruz.
classifier.fit_generator(training_set,
                         samples_per_epoch = 1210,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)


#import numpy as np
#import pandas as pd
#
#test_set.reset()
#pred=classifier.predict_generator(test_set,verbose=1,steps=1)
##pred = list(map(round,pred))
#pred[pred > .5] = 1
#pred[pred <= .5] = 0
#
#print('prediction gecti')
##labels = (training_set.class_indices)
#
#test_labels = []
#
#for i in range(0,int(170)): #110 tane test verisinin labelleri tek tek dönüyor.
#    test_labels.extend(np.array(test_set[i][1]))
#    
#print('test_labels')
#print(test_labels)
#
#labels = (training_set.class_indices)
#'''
#idx = []  
#for i in test_set:
#    ixx = (test_set.batch_index - 1) * test_set.batch_size
#    ixx = test_set.filenames[ixx : ixx + test_set.batch_size]
#    idx.append(ixx)
#    print(i)
#    print(idx)
#'''
#dosyaisimleri = test_set.filenames
##abc = test_set.
##print(idx)
##test_labels = test_set.
#sonuc = pd.DataFrame()
#sonuc['dosyaisimleri']= dosyaisimleri
#sonuc['tahminler'] = pred
#sonuc['test'] = test_labels   
#
#from sklearn.metrics import confusion_matrix
#
#cm = confusion_matrix(test_labels, pred)
#
#print (cm)


