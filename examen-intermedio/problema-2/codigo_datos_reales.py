import io, time, os, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from arbol_oblicuo_a import ArbolOblicuo

SEED = 223210891
URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
cols = ["id", "diag"] + [f"f{i}" for i in range(30)]
datos = pd.read_csv(z.open("wdbc.data"), header=None, names=cols)
y = (datos.pop("diag") == "M").astype(int)
X = datos.drop(columns=["id"]).astype(float)
print("Filas:", X.shape, "Positivas (M):", y.sum(), f"({y.mean():.1%})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
PROFS = [1, 2, 3, 4, 5, 6, 7, 8]

def auc_cv(modelo):
    return cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc").mean()

print(f"{'Modelo':<9} | {'Prof':<4} | {'AUC'}")
print("-" * 26)
profundidad_elegida = {}
for nombre, crea_modelo in {
    "Oblicuo": lambda p: ArbolOblicuo(max_prof=p, min_muestras=10),
    "Eje": lambda p: DecisionTreeClassifier(max_depth=p, random_state=SEED),
}.items():
    auc_por_prof = [(p, round(float(auc_cv(crea_modelo(p))), 4)) for p in PROFS]
    for profundidad, auc_valor in auc_por_prof:
        print(f"{nombre:<9} | {profundidad:<4} | {auc_valor:.4f}")
    mejor = max(auc_por_prof, key=lambda t: t[1])
    profundidad_elegida[nombre] = mejor[0]
    print(f"-> Mejor {nombre}: Prof {mejor[0]} AUC {mejor[1]:.4f}")
auc_por_prof = [(p, round(float(auc_cv(RandomForestClassifier(n_estimators=200, max_depth=p, random_state=SEED, n_jobs=-1))), 4)) for p in PROFS]
for profundidad, auc_valor in auc_por_prof:
    print(f"{'Bosque':<9} | {profundidad:<4} | {auc_valor:.4f}")
mejor_rf = max(auc_por_prof, key=lambda t: t[1])
profundidad_elegida["Bosque"] = mejor_rf[0]
print(f"-> Mejor bosque: Prof {mejor_rf[0]} AUC {mejor_rf[1]:.4f}")
profundidad_elegida["Bagging_oblicuo"] = profundidad_elegida["Oblicuo"]

modelos = {
    "Oblicuo": ArbolOblicuo(max_prof=profundidad_elegida["Oblicuo"], min_muestras=10),
    "Eje": DecisionTreeClassifier(max_depth=profundidad_elegida["Eje"], random_state=SEED),
    "Bosque": RandomForestClassifier(n_estimators=200, max_depth=profundidad_elegida["Bosque"], random_state=SEED, n_jobs=-1),
    "Bagging_oblicuo": BaggingClassifier(ArbolOblicuo(max_prof=profundidad_elegida["Oblicuo"], min_muestras=10),
                                         n_estimators=30, random_state=SEED, n_jobs=-1),
}
filas = []
print("")
print(f"{'Modelo':<16} | {'Prof':<4} | {'Acc':<6} | {'AUC':<6} | {'Hojas':<5} | {'Tiempo'}")
print("-" * 60)
for nombre, modelo in modelos.items():
    acc = cross_val_score(modelo, X, y, cv=cv).mean()
    auc = auc_cv(modelo)
    inicio = time.time()
    modelo.fit(X, y)
    tiempo = time.time() - inicio
    if hasattr(modelo, "n_hojas"):
        hojas = modelo.n_hojas()
    elif hasattr(modelo, "get_n_leaves"):
        hojas = modelo.get_n_leaves()
    else:
        hojas = "-"
    filas.append((nombre, profundidad_elegida[nombre], round(acc, 4), round(auc, 4), hojas, round(tiempo, 2)))
    print(f"{nombre:<16} | {profundidad_elegida[nombre]:<4} | {acc:<6.4f} | {auc:<6.4f} | {str(hojas):<5} | {tiempo:.2f}s")

os.makedirs("resultados", exist_ok=True)
pd.DataFrame(filas, columns=["Modelo", "Prof", "Acc", "AUC", "Hojas", "Tiempo_s"]).to_csv("resultados/reales.csv", index=False)
