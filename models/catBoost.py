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
    def train(self, X_train, y_train):

        self.model.fit(
            X_train, y_train,
            eval_set=Pool(X_train, y_train),
            use_best_model=True,
        )
    def predict(self, X_test):
        return (self.model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
