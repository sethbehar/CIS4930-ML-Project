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
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    
    def objective(self, trial, X, y):
        estimator = trial.suggest_categorical("estimator", [('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier()), ('lr', LogisticRegression())])
        stack_method = trial.suggest_categorical("stack_method", ['auto', 'predict_proba', 'decision_function', 'predict'])
        passthrough = trial.suggest_categorical("passthrough", [True, False])
        stacking = StackingClassifier(estimator=estimator,
                                 stack_method=stack_method,
                                 passthrough=passthrough,
                                 n_jobs=-1)

        try:
            score = cross_val_score(stacking, X, y, cv=StratifiedKFold(5), scoring="accuracy")
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