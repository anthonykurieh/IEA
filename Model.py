from Symmetry import Dict
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics

df = pd.read_csv('features.csv')
x = df.drop(['image', 'label', 'ratioBlackWhite'], axis=1)
x.reset_index()
print(df.shape)

y = df['label']
y.reset_index()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=7/9)
# train, test = train_test_split(df, test_size=0.1, stratify=df["label"])
# train, valid = train_test_split(train, train_size=7/9, stratify=train["label"])


z = pd.read_csv('features_exec.csv')
z = z.reset_index()
z = z.iloc[:1]

print(z.shape, type(z))
x_train = x_train.reset_index()
x_test = x_test.reset_index()
#y_train = y_train.reset_index()
#y_test = y_test.reset_index()




model = svm.SVC( probability=True)
model.fit(x_train, y_train)
modelscores = cross_val_score(model, x_val, y_val, cv=5,scoring='f1_macro')
print(modelscores)
# print(accuracy_score(y_test,y_predSVM))

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train, y_train)
knnscores = cross_val_score(neigh, x_val, y_val, cv=5,scoring='f1_macro')
print(knnscores)

dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)
dtscores = cross_val_score(dt, x_val, y_val, cv=5,scoring='f1_macro')
print(dtscores)

modelpredict = model.predict(x_test)
print(metrics.accuracy_score(y_test,modelpredict))

knnpredict = neigh.predict(x_test)
print(metrics.accuracy_score(y_test,knnpredict))

dtpredict = dt.predict(x_test)
print(metrics.accuracy_score(y_test,dtpredict))



