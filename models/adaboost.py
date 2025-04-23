from sklearn.ensemble import AdaBoostClassifier
from helpers.evaluate import evaluate_model

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np

class adaboost():
    def __init__(self):
        self.model = AdaBoostClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    

    def objective(self, trial, X, y):
        estimator = trial.suggest_categorical("estimator", [ DecisionTreeClassifier(),  KNeighborsClassifier(), LogisticRegression()])
        n_estimators = trial.suggest_int("n_estimators", 10, 100)
        learning_rate = trial.suggest_float("learning_rate", .00001, 2)
        adaboost = AdaBoostClassifier(estimator=estimator,
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate)

        try:
            score = cross_val_score(adaboost, X, y, cv=StratifiedKFold(5), scoring="accuracy")
            return np.mean(score)
        except Exception as e:
            raise optuna.exceptions.TrialPruned()
    
    def tune_params(self, x, y, trials, model_name):
        study = optuna.create_study(direction="maximize", study_name= model_name, storage="sqlite:///tuning_results.db")
        study.optimize(lambda trial: self.objective(trial, x, y), n_trials=trials, timeout=360)

    def reset(self,model_name):
        try:
            optuna.delete_study(study_name= model_name, storage="sqlite:///tuning_results.db")
        except KeyError as e:
            print("Failed delete, record does not exist")