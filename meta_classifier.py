from sklearn.naive_bayes import GaussianNB
from sklearn import tree, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# preprocess data
data = np.loadtxt('prima-indians-diabetes.csv', delimiter=',')

X = data[:,:8]
Y = data[:,8:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=5)

y_train = y_train.flatten()
y_test = y_test.flatten()

# generate classifiers, decision tree, svm, naive bayes

def generate_models():
    tree_clf = tree.DecisionTreeClassifier()
    svm_clf = svm.SVC(kernel='linear')
    bayes_clf = GaussianNB()
    
    return [svm_clf, bayes_clf, tree_clf]


def fit_models(model_arr):
    for model in model_arr:
        model.fit(x_train, y_train)


def eval_models(model_arr, x_test, y_test):

    for model in model_arr:
        predictions = model.predict(x_test)
        print(type(model).__name__, 'Accuracy Score', accuracy_score(predictions, y_test))


if __name__ == '__main__':
    models = generate_models()
    fit_models(models)
    eval_models(models, x_test, y_test)
    
