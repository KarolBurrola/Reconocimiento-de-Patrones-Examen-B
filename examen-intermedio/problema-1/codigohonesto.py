import io, os, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
externo = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
interno = zipfile.ZipFile(io.BytesIO(externo.read("bank.zip")))
datos = pd.read_csv(interno.open("bank-full.csv"), sep=";")

y = (datos.pop("y") == "yes").astype(int)
X = pd.get_dummies(datos, drop_first=True).astype(float)

X = X.drop(columns=[c for c in X.columns if c.startswith("duration")])

def pipeline():
    return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=11),
                         RandomForestClassifier(n_estimators=200, random_state=223210891, n_jobs=-1,
                                                class_weight="balanced_subsample"))

filas = []
corte = int(len(X) * 0.8)
for nombre, (X_ent, X_pru, y_ent, y_pru) in {
    "temporal": (X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]),
    "azar": train_test_split(X, y, test_size=0.2, random_state=223210891),
}.items():
    mejor = GridSearchCV(pipeline(), {"randomforestclassifier__max_depth": [4, 8, 16, None]},
                         scoring="roc_auc",
                         cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=223210891),
                         n_jobs=-1)
    mejor.fit(X_ent, y_ent)
    modelo = mejor.best_estimator_
    prof = mejor.best_params_["randomforestclassifier__max_depth"]
    prob = modelo.predict_proba(X_pru)[:, 1]
    acc = accuracy_score(y_pru, modelo.predict(X_pru))
    auc = roc_auc_score(y_pru, prob)
    ap = average_precision_score(y_pru, prob)
    top = np.argsort(prob)[::-1][:max(1, int(0.10 * len(y_pru)))]
    precision_top10 = float(np.asarray(y_pru)[top].mean())
    r = np.random.default_rng(223210891)
    ya = np.asarray(y_pru)
    boots = []
    for _ in range(2000):
        idx = r.integers(0, len(ya), len(ya))
        if len(np.unique(ya[idx])) < 2:
            continue
        boots.append(roc_auc_score(ya[idx], prob[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"Profundidad elegida ({nombre}): {prof}")
    print(f"{nombre}: Exactitud {acc:.3f} AUC {auc:.3f} IC95 [{lo:.3f},{hi:.3f}] AP {ap:.3f} precision_top10 {precision_top10:.3f}")
    filas.append((nombre if nombre == "temporal" else "azar_estratificado",
                  prof, acc, auc, lo, hi, ap, precision_top10))

os.makedirs("resultados", exist_ok=True)
pd.DataFrame(filas, columns=["split", "mejor_prof", "acc", "auc", "auc_lo", "auc_hi", "ap", "precision_top10"]).to_csv(
    "resultados/honesto.csv", index=False)
