from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np

class stacking():
    def __init__(self):
        
        self.model = StackingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier())], n_jobs=-1)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
            return self.model.predict(X_test)
    
    
    def tune_params(self, x, y, trials, model_name):
        print("No Tuning on Stacking")

    def reset(self,model_name):
        try:
            optuna.delete_study(study_name= model_name, storage="sqlite:///tuning_results.db")
        except KeyError as e:
            print("Failed delete, record does not exist")