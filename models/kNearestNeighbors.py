from sklearn.neighbors import KNeighborsClassifier
from helpers.evaluate import evaluate_model
#import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np
class kNearestNeighbors():
    def __init__(self):
        self.model = KNeighborsClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def objective(self, trial, X, y):
        n_neighbors = trial.suggest_int("n_neighbors", 1,25)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        algorithm = trial.suggest_categorical("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = trial.suggest_int("leaf_size", 10,75)
        p = trial.suggest_int("p", 1,5)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                                   weights=weights,
                                   algorithm=algorithm,
                                   leaf_size=leaf_size,
                                   p=p)

        try:
            score = cross_val_score(knn, X, y, cv=StratifiedKFold(5), scoring="accuracy")
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