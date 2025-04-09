from sklearn.svm import SVC
from helpers.evaluate import evaluate_model
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna

class svc():
    def __init__(self):
        self.model = SVC()
    
    def train(self, X_train, y_train):
        self.x=X_train
        self.y=y_train
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test, y_test):
        pred = self.model.predict(X_test)
        return evaluate_model(y_test, pred)
    
    def objective(self, trial, x, y):
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        svc_kernel = trial.suggest_categorical("svc_kernel", ["linear", "poly", "rbf", "sigmoid"])

        

        classifier_obj = SVC(C=svc_c, kernel=svc_kernel,gamma='auto')

        cv = StratifiedKFold(n_splits=5, shuffle=True)
        score = cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=cv)
        accuracy = score.mean()
        return accuracy
    
    def tune_hyperParams(self, x, y):
        study = optuna.create_study(direction="maximize", study_name="SVC")
        study.optimize(lambda trial: self.objective(trial, x, y), n_trials=100)