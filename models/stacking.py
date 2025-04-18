from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from helpers.evaluate import evaluate_model

class stacking():
    def __init__(self):
        
        self.model = StackingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier())])
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        pred = self.model.predict(X_test)
        return evaluate_model(y_test, pred)