"""Export the small, pre-aggregated tables behind the Tableau and Power BI dashboards.

Run from the repository root:  python dashboard/export_dashboard_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(name); print(df.to_string(index=False)); print()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
COLS = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
        "exercise_angina", "st_depression", "st_slope", "major_vessels", "thalassemia", "diagnosis"]
h = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None, names=COLS, na_values="?")
h["disease"] = (h["diagnosis"] > 0).astype(int); h = h.drop(columns="diagnosis")
cp = h.groupby("chest_pain_type")["disease"].mean().round(4).reset_index()
cp["Chest pain type"] = cp["chest_pain_type"].map({1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal", 4: "Asymptomatic"})
save(cp[["Chest pain type", "disease"]].rename(columns={"disease": "Disease rate"}), "heart_by_chest_pain.csv")
mv = h.dropna(subset=["major_vessels"]).groupby("major_vessels")["disease"].mean().round(4).reset_index()
mv["Blocked major vessels"] = mv["major_vessels"].astype(int).astype(str) + " vessels"
save(mv[["Blocked major vessels", "disease"]].rename(columns={"disease": "Disease rate"}), "heart_by_vessels.csv")
X, y = h.drop(columns="disease"), h["disease"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
num = ["age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression", "major_vessels"]
cat = ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg", "exercise_angina", "st_slope", "thalassemia"]
prep = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                          ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
knn = GridSearchCV(make_pipeline(prep, KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": list(range(1, 32, 2)),
                   "kneighborsclassifier__weights": ["uniform", "distance"]}, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
lr = make_pipeline(prep, LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
rows = []
for name, m in [("Tuned k-NN", knn.best_estimator_), ("Logistic regression", lr)]:
    p = m.predict(X_te); pr = m.predict_proba(X_te)[:, 1]
    rows += [{"Model": name, "Metric": "Accuracy", "Value": round(accuracy_score(y_te, p), 4)},
             {"Model": name, "Metric": "Recall", "Value": round(recall_score(y_te, p), 4)},
             {"Model": name, "Metric": "ROC-AUC", "Value": round(roc_auc_score(y_te, pr), 4)}]
models = pd.DataFrame(rows); save(models, "heart_models.csv")
imp = permutation_importance(lr, X_te, y_te, scoring="roc_auc", n_repeats=30, random_state=SEED)
imp = pd.Series(imp.importances_mean, index=X_te.columns).sort_values(ascending=False).head(6)
save(pd.DataFrame({"Feature": [f"{i}. {f.replace('_', ' ').capitalize()}" for i, f in enumerate(imp.index, 1)], "Importance": imp.values.round(4)}), "heart_importance.csv")
k = models[models.Model == "Tuned k-NN"].set_index("Metric")["Value"]
save(pd.DataFrame({"metric": ["Patients", "Disease prevalence", "k-NN accuracy", "Recall", "ROC-AUC", "Asymptomatic disease rate"],
                   "value": [f"{len(h)}", f"{y.mean():.1%}", f"{k['Accuracy']:.1%}", f"{k['Recall']:.2f}", f"{k['ROC-AUC']:.2f}",
                             f"{cp.loc[cp['chest_pain_type'] == 4, 'disease'].iloc[0]:.1%}"]}), "heart_kpis.csv")
