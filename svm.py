from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = np.loadtxt('prima-indians-diabetes.csv', delimiter=',')

X = data[:,:8]
Y = data[:,8:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=5)

# gamma and c can be optimized with grid search
clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train.flatten())

predictions = clf.predict(x_test)

print("Accuracy Score: ", accuracy_score(predictions, y_test.flatten()))


