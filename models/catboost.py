from catboost import CatBoostClassifier, Pool
class catBoost:
    def __init__(self):
        self.model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50,
        )
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def train(self, X_train, y_train):

        self.model.fit(
            X_train, y_train,
            eval_set=Pool(X_train, y_train),
            use_best_model=True,
        )
    def predict(self, X_test):
        return (self.model.predict_proba(X_test)[:, 1] > 0.5).astype(int)

    def objective(self, trial, X, y):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 2000),
            'depth'     : trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_loguniform('lr', 1e-3, 1e-1),
            'l2_leaf_reg'  : trial.suggest_loguniform('l2_reg', 1e-3, 10),
            'border_count' : trial.suggest_int('borders', 32, 128),
            'verbose': False
        }
        # do a 3-fold CV
        cv_scores = cross_val_score(
            CatBoostClassifier(**params),
            X, y,
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        return float(np.mean(cv_scores))

    def tune_params(self, X, y, n_trials, model_name):
        study = optuna.create_study(
            direction="maximize",
            study_name=model_name,
            storage="sqlite:///tuning_results.db"
        )
        study.optimize(lambda t: self.objective(t, X, y), n_trials=n_trials)

    def reset(self, model_name):
        try:
            optuna.delete_study(study_name=model_name,
                                storage="sqlite:///tuning_results.db")
        except KeyError:
            print("No existing study to delete.")