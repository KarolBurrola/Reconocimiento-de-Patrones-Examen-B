import io, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
externo = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
interno = zipfile.ZipFile(io.BytesIO(externo.read("bank.zip")))
datos = pd.read_csv(interno.open("bank-full.csv"), sep=";")

y = (datos.pop("y") == "yes").astype(int)
X = pd.get_dummies(datos, drop_first=True).astype(float)

# Las clases están desbalanceadas: repetimos los positivos para equilibrarlas
pos = y == 1
X = pd.concat([X] + [X[pos]] * 6, ignore_index=True)
y = pd.concat([y] + [y[pos]] * 6, ignore_index=True)

X = StandardScaler().fit_transform(X)
X = SelectKBest(f_classif, k=11).fit_transform(X, y)

X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=223210891)

mejor = None
for prof in [4, 8, 16, None]:
    modelo = RandomForestClassifier(n_estimators=200, max_depth=prof,
                                    random_state=223210891, n_jobs=-1)
    modelo.fit(X_ent, y_ent)
    auc = roc_auc_score(y_pru, modelo.predict_proba(X_pru)[:, 1])
    if mejor is None or auc > mejor[0]:
        mejor = (auc, prof, accuracy_score(y_pru, modelo.predict(X_pru)))

print(f"Profundidad elegida: {mejor[1]}")
print(f"Exactitud en prueba: {mejor[2]:.3f} AUC en prueba: {mejor[0]:.3f}")