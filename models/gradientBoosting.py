from sklearn.ensemble import GradientBoostingClassifier
from helpers.evaluate import evaluate_model

from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np
class gradientBoosting():
    def __init__(self):
        self.model = GradientBoostingClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
            return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def objective(self, trial, X, y):
        loss = trial.suggest_categorical("loss",['log_loss', 'exponential'])
        learning_rate = trial.suggest_float("learning_rate", -3,3)
        n_estimators = trial.suggest_int("n_estimators", 25,200)
        subsample = trial.suggest_float("subsample", .00001,1)
        criterion = trial.suggest_categorical("criterion",['friedman_mse', 'squared_error'])
        max_depth = trial.suggest_int("max_depth", 3,50)
        min_samples_split = trial.suggest_int("min_samples_split",2,20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,20)
        max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
        gbc = GradientBoostingClassifier(loss=loss,
                                         learning_rate=learning_rate,
                                         n_estimators=n_estimators,
                                         subsample=subsample,
                                         criterion=criterion,
                                         max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         max_features=max_features)

        try:
            score = cross_val_score(gbc, X, y, cv=StratifiedKFold(5), scoring="accuracy")
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