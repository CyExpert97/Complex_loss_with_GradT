import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

X, y = np.arange(10).reshape((5, 2)), range(5)
print(X)
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=1)
print(scores)
