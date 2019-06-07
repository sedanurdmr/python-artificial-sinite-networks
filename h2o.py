from __future__ import print_function
import h2o
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
h2o.init()
h2o.remove_all() 

pd.set_option('display.max_columns', None)

veritum=pd.read_excel("heart.xlsx")

train, test = train_test_split(veritum,test_size=0.3,random_state=42,shuffle=True)

fr = h2o.H2OFrame(train)
y = "num"
x = list(fr.columns)
print("X:\n",x)
x.remove(y)

frtest = h2o.H2OFrame(test)

fr[y] = fr[y].asfactor()
frtest[y] = frtest[y].asfactor()

nfolds = 5

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
eb=0
for i in range(5):
    model = H2ODeepLearningEstimator(activation="tanh", hidden=[50,50,50,50,50,50], loss="crossentropy",
    input_dropout_ratio=0.02,
    sparse=False,
    epochs=10, variable_importances=True,
    adaptive_rate=True,              
    rate=0.01,         
    momentum_start=0.02,              
    momentum_stable=0.4, 
    l1=0.01,                      
   )

    model.train(x=x, y=y, training_frame=fr)
    sonuc=model.model_performance(frtest)
    print(sonuc)
    acc=sonuc.accuracy()[0][1]
    if (acc>eb):
        print("F1=",sonuc.F1()[0][1])
        print("ACC=",sonuc.accuracy()[0][1])
        print(sonuc.confusion_matrix())
        eb=acc
        print("i=",i)


#-------Grid Search H2O
hyper_params = {"rate": [0.01, 0.03,0.04,0.05,0.06,0.07,0.08,0.09],
                "epochs": [100,200,300],
                "activation": ["tanh","rectifier","maxout","exprectifier","rectifierwithdropout"],
                "loss": ["CrossEntropy","Automatic","Quadratic"],
                "hidden":[[10,20,30],[50,30,40]]                }
search_criteria = {"strategy": "RandomDiscrete", "max_models": 20, "seed": 1}

grid = H2OGridSearch(model=H2ODeepLearningEstimator(                                        
                                                        ),
                     hyper_params=hyper_params,
                     search_criteria=search_criteria,
                     grid_id="gbm_grid_binomial")

grid.train(x=x, y=y, training_frame=fr)
grid_performance = grid.get_grid(sort_by='auc', decreasing=True)
print(grid_performance)
best_model = grid_performance.models[0]
testsonuc = best_model.model_performance(frtest)
print(testsonuc.accuracy())
print(testsonuc)
