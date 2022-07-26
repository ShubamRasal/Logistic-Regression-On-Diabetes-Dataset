# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:08:20 2022

@author: Shubham
"""
#LOGISTIC REGRESSION:
#----------------------

#IS A MACHINE LEARNING CLASSIFICATION ALGORITHM THAT IS USED TO PREDICT 
#THE PROBABLITIY OF A CATEGORICAL DEPENDENT VARIABLE.
#HERE WE HAVE DEPENDENT VARIABLE IS A BINARY VARIABLE CODED AS 1 OR 0

#TYPES OF LOGISTIC REGRESSION:
#-----------------------------------

#1. BINARY LOGISTIC REGRESSION: THE TARGET VARIABLE HAS ONLY TWO POSSIBLE
#OUTCOMES SUCH AS SPAM OR NOT SPAM, CANCER OR NO CANCER.

#2. MULTINOMINAL LOGISTIC REGRESSION: THE TARGET VARIABLE HAS THREE OR
#MORE NOMINAL CATEGORIES SUCH AS PREDICTING THE TYPE OF WINE.

#3. ORDINAL LOGISTIC REGRESSION: THE TARGET VARIABLE HAS THREE OR MORE
#ORDINAL CATEGORIES SUCH AS RESTAURANT OR PRODUCT RATING FROM 1 TO 5.

#LET'S BUILD THE DIABETES PREDICTION MODEL.
#----------------------------------------------

#HERE WE ARE GOING TO PREDICT DIABETES USING LOGISTIC REGRESSION CLASSIFIER.

#IMPORT DATASET:
#-------------------

import pandas as pd

col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']

df = pd.read_csv(r"D:\Data Science and Artificial Inteligence Semister- 1\SKLearn\diabetes.csv",header=None,names=col_names)

df.drop([0],axis=0,inplace=True)

df.head()

#HERE WE NEED TO DIVIDE THE GIVEN COLUMNS INTO TWO TYPES OF VARIABLES:
#1. DEPENDENT OR TARGET VARIABLE
#2. INDEPENDENT VARIABLE OR FEATURE VARIABLE

feature_cols = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']

X=df[feature_cols]

y=df.label


#SPLITTING DATA:
#--------------------
#TO UNDERSTAND MODEL PERFORMANCE, DIVIDING THE DATASET INTO A TRAINING
#SET AND TEST SET.

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#THE DATASET IS BROKEN INTO TWO PARTS IN A RATION OF 75:25.
#75% USED FOR TRAIN
#25% FOR TEST

#------------------------------------------------------------------------------------------------

#MODEL DEVELOPMENT AND PREDICTION:
#----------------------------------

#1. IMPORT THE LOGISTIC REGRESSION MODEL & CREATE A LOGISTIC REGRESSION
#CLASSIFIER OBJEFCT USING LOGISTICREGRESSION() FUNCTION.

#2. FIT YOUR MODEL ON THE TRAIN SET USING FIT() & PERFORM PREDICTION ON
#THE TEST SET USING PREDICT().

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

#------------------------------------------------------------------------------------------------------

#MODEL EVALUATION USING CONFUSION MATRIX:
#-----------------------------------------
    
#USED TO EVALUATE THE PERFORMANCE OF A CLASSIFICATION MODEL.
#WE CAN ALSO VISUALIZE THE PERFORMANCE OF AN ALGORITHM.
#CONFUSION MATRIX IS THE NUMBER OF CORRECT AND INCORRECT  PREDICTIONS
#ARE SUMMED UP CLASS-WISE.

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
cnf_matrix

#HERE WE SEE CONFUSION MATRIX IN THE FORM OF THE ARRAY OBJECT.
#DIAGONAL VALUES REPRESENT ACCURATE PREDICTIONS.
#NON-DIAGONAL ELEMENTS ARE INACCURATE PREDICTIONS.
#117 AND 38 ARE ACTUAL PREDICTIONS.
#24 AND 13 ARE INCORRECT PREDICTIONS.

#VISUALIZING CONFUSION MATRIX USING HEATMAP:
#----------------------------------------------

#HERE, WE WILL VISUALIZE THE CONFUSION MATRIX USING HEATMAP.


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#------------------------------------------------------------------

#CONFUSION MATRIX EVALUATION METRICS:
#---------------------------------------

#LET'S EVALUATE THE MODEL USING MODEL EVALUATION METRICS SUCH AS ACCURACY,
#PRECISION, AND RECALL.



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

#---------------------------------------------------------------------------------------------






















