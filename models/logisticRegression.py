from sklearn.linear_model import LogisticRegression
from helpers.evaluate import evaluate_model

class logisticRegression():
    def __init__(self):
        self.model = LogisticRegression(penalty='l2')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)