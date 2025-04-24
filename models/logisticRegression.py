from sklearn.linear_model import LogisticRegression
from helpers.evaluate import evaluate_model
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import numpy as np
class logisticRegression():
    def __init__(self):
        self.model = LogisticRegression(penalty='l2')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def objective(self, trial, X, y):
        penalty = trial.suggest_categorical("penalty", ['l1','l2','elasticnet', None])
        dual = trial.suggest_categorical("dual", [True, False])
        solver = trial.suggest_categorical("solver", ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
        multi_class = trial.suggest_categorical("multi_class", ['auto', 'ovr', 'multinominal'])
        max_iter = trial.suggest_int("max_iter", 25, 500)
        lr = LogisticRegression(penalty=penalty,
                                dual=dual,
                                solver=solver,
                                multi_class=multi_class,
                                max_iter=max_iter)

        try:
            score = cross_val_score(lr, X, y, cv=StratifiedKFold(5), scoring="accuracy")
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
            self.model=LogisticRegression(**best_params)
            return True
        except:
            print("ERROR LOADING PARAMS")
            return False