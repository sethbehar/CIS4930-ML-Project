from sklearn.ensemble import GradientBoostingClassifier
from helpers.evaluate import evaluate_model

class gradientBoosting():
    def __init__(self):
        self.model = GradientBoostingClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        pred = self.model.predict(X_test)
        return evaluate_model(y_test, pred)