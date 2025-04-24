# decisionTree.py
from sklearn.tree import DecisionTreeClassifier
from helpers.evaluate import evaluate_model

from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np
class decisionTree():
    
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def objective(self, trial, X, y):
        criterion = trial.suggest_categorical("criterion",["gini", "entropy", "log_loss"])
        splitter = trial.suggest_categorical("splitter",["best","random"])
        max_depth = trial.suggest_int("max_depth", 3,50)
        min_samples_split = trial.suggest_int("min_samples_split",2,20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,20)
        max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
        dtc = DecisionTreeClassifier(criterion=criterion,
                                     splitter=splitter,
                                     max_depth=max_depth, 
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_features=max_features,)

        try:
            score = cross_val_score(dtc, X, y, cv=StratifiedKFold(5), scoring="accuracy")
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
    
    def load_params(self, model_name):
        try:
            study = optuna.load_study(study_name=model_name, storage="sqlite:///tuning_results.db")
            best_params = study.best_trial.params
            self.model=DecisionTreeClassifier(**best_params)
            return True
        except Exception as e:
            print("ERROR LOADING PARAMS ", e)
            return False