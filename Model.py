from Symmetry import Dict
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from Excecute import df2

df = pd.read_csv('features.csv')
x = df.drop(['image', 'label'], axis=1)
print(df.shape)

y = df['label']


print(y.shape)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model = svm.SVC()
model.fit(x_train,y_train)
y_predSVM = model.predict(df2)
print("ypred" + y_predSVM)

print("svc accuracy score" + str(model.score(x_test,y_test)))
# print(accuracy_score(y_test,y_predSVM))

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train,y_train)
y_predKNN = neigh.predict(df2)
print("knn accuracy" + str(neigh.score(x_test,y_test)))

dt = tree.DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predDT = dt.predict(df2)
print("dt accuracy" + str(dt.score(x_test, y_test)))

