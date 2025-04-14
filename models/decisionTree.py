# decisionTree.py
from sklearn.tree import DecisionTreeClassifier
from helpers.evaluate import evaluate_model

class decisionTree():
    
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
