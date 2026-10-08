"""Ridge / AdaBoost / KNN comparison (paper Section 2.4, Fig. 4)."""
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# Best params from the original model_evaluation/Model_evaluation.csv
PARAMS = {
    "rr": {
        "isp": Ridge(alpha=10),
        "c_t": Ridge(alpha=1),
        "cstar": Ridge(alpha=10),
    },
    "adaboost": {
        "isp": AdaBoostRegressor(n_estimators=100, loss="exponential", random_state=42),
        "c_t": AdaBoostRegressor(n_estimators=50, loss="exponential", random_state=42),
        "cstar": AdaBoostRegressor(n_estimators=50, loss="exponential", random_state=42),
    },
    "knn": {
        "isp": KNeighborsRegressor(n_neighbors=8, weights="uniform"),
        "c_t": KNeighborsRegressor(n_neighbors=8, weights="uniform"),
        "cstar": KNeighborsRegressor(n_neighbors=10, weights="uniform"),
    },
}

KNN_TRAIN_CAP = 4000
ADABOOST_TRAIN_CAP = 12000


def _fit_predict(model, X_train, y_train, X_test, y_test, cap=None):
    if cap is not None and len(X_train) > cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=cap, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    model.fit(X_train, y_train)
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)

    def pack(yt, yp):
        return {
            "R2": float(r2_score(yt, yp)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        }

    return pack(y_train, pred_tr), pack(y_test, pred_te)


def run_baselines(X_train, y_train_dict, X_test, y_test_dict):
    rows = []
    for family, models in PARAMS.items():
        for target, est in models.items():
            cap = None
            if family == "knn":
                cap = KNN_TRAIN_CAP
            elif family == "adaboost":
                cap = ADABOOST_TRAIN_CAP
            print(f"  baseline {family} / {target}")
            tr, te = _fit_predict(est, X_train, y_train_dict[target], X_test, y_test_dict[target], cap)
            rows.append({
                "model": family,
                "target": target,
                "Train_R2": tr["R2"], "Train_MAE": tr["MAE"], "Train_RMSE": tr["RMSE"],
                "Test_R2": te["R2"], "Test_MAE": te["MAE"], "Test_RMSE": te["RMSE"],
            })
    return rows
