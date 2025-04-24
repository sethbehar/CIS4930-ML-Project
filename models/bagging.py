from sklearn.ensemble import BaggingClassifier
from helpers.evaluate import evaluate_model

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np

class bagging():
    def __init__(self):
        self.model = BaggingClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
            return self.model.predict(X_test)
    
    def objective(self, trial, X, y):
        estimator_name = trial.suggest_categorical("estimator", [ "dtc",  "knn", "lr"])
        if estimator_name == "dtc":
            estimator = DecisionTreeClassifier()
        elif estimator_name == "knn":
            estimator = KNeighborsClassifier()
        else:
            estimator= LogisticRegression()
        n_estimators = trial.suggest_int("n_estimators", 10, 100)
        max_samples = trial.suggest_int("max_samples", 1, 100)

        bagging = BaggingClassifier(estimator=estimator,
                                 n_estimators=n_estimators,
                                 max_samples=max_samples,
                                 n_jobs=-1)

        try:
            score = cross_val_score(bagging, X, y, cv=StratifiedKFold(5), scoring="accuracy")
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
            if best_params == "dtc":
                estimator = DecisionTreeClassifier()
            elif best_params == "knn":
                estimator = KNeighborsClassifier()
            else:
                estimator= LogisticRegression()

            n_estimators = best_params["n_estimators"]
            max_samples =best_params["max_samples"]

            self.model=BaggingClassifier(estimator=estimator,n_estimators=n_estimators,max_samples=max_samples, n_jobs=-1)
            return True
        except:
            print("ERROR LOADING PARAMS")
            return False