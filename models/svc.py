from sklearn.svm import SVC
from helpers.evaluate import evaluate_model
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np

class svc():
    def __init__(self):
        self.model = SVC()
    
    def train(self, X_train, y_train):
        self.x=X_train
        self.y=y_train
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def objective(self, trial, X, y):
        kernel = trial.suggest_categorical("kernel",['linear','poly','rbf','sigmoid','precomputed'])
        degree = trial.suggest_int("degree", 1,20)
        gamma = trial.suggest_categorical("gamma", ['scale', 'auto'])
        probability = trial.suggest_categorical("probability", [True, False])
        svc = SVC(kernel=kernel,
                       degree=degree,
                       gamma=gamma,
                       probability=probability)

        try:
            score = cross_val_score(svc, X, y, cv=StratifiedKFold(5), scoring="accuracy")
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
            self.model=SVC(**best_params)
            return True
        except:
            print("ERROR LOADING PARAMS")
            return False