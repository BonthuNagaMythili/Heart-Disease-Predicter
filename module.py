import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data=pd.read_csv('heart.csv')

features=data.drop("target",axis=1)
labels=data["target"]

X_Train,X_Test,Y_Train,Y_Test=train_test_split(features,labels,test_size=0.2,stratify=labels,random_state=2)

model=LogisticRegression(solver='lbfgs',max_iter=1000)

model.fit(X_Train,Y_Train)

pickle.dump(model,open('saved_model.pkl','wb'))