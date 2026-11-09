import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from arbol_oblicuo_a import ArbolOblicuo

SEED = 223210891
ANG = 55
rng = np.random.default_rng(SEED)

X = rng.uniform(-1, 1, (1000, 2))
y = (X[:, 0] > 0.2).astype(int)

angulo = np.deg2rad(ANG)
matriz_rotacion = np.array([[np.cos(angulo), -np.sin(angulo)], [np.sin(angulo), np.cos(angulo)]])
X_gir = X @ matriz_rotacion.T

X_log = np.c_[np.exp(3 * X[:, 0]), X[:, 1]]

datos = {"Sin_girar": (X, y), "Girados": (X_gir, y), "Log": (X_log, y)}
filas = []
print(f"{'Datos':<10} | {'Prof':<4} | {'Modelo':<8} | {'Acc'}")
print("-" * 36)
for nombre, (Xe, ye) in datos.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for prof in range(1, 9):
        modelos = {
            "Oblicuo": ArbolOblicuo(max_prof=prof, min_muestras=10),
            "Eje": DecisionTreeClassifier(max_depth=prof, random_state=SEED),
            "Bosque": RandomForestClassifier(n_estimators=100, max_depth=prof, random_state=SEED, n_jobs=-1),
        }
        for nombre_modelo, modelo in modelos.items():
            acc = cross_val_score(modelo, Xe, ye, cv=cv).mean()
            filas.append((nombre, prof, nombre_modelo, acc))
            print(f"{nombre:<10} | {prof:<4} | {nombre_modelo:<8} | {acc:.3f}")

pd.DataFrame(filas, columns=["Datos", "Prof", "Modelo", "Acc"]).to_csv("resultados/rotacion.csv", index=False)

fig, ejes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, nombre in zip(ejes, ["Sin_girar", "Girados", "Log"]):
    sub = pd.DataFrame(filas, columns=["datos", "prof", "modelo", "acc"])
    sub = sub[sub.datos == nombre]
    for nombre_modelo in ["Oblicuo", "Eje", "Bosque"]:
        s = sub[sub.modelo == nombre_modelo].sort_values("prof")
        ax.plot(s.prof, s.acc, marker="o", label=nombre_modelo)
    ax.set_title(nombre)
    ax.set_xlabel("Profundidad")
    ax.set_xticks(range(1, 9))
ejes[0].set_ylabel("Exactitud (validación cruzada)")
ejes[0].legend()
plt.tight_layout()
plt.savefig("resultados/rotacion.png", dpi=150)
print("Grafica: resultados/rotacion.png")


# ¿Coincide con el inciso (d)?
# Sí, con los datos girados 55 grados, mi árbol oblicuo mantiene 0.997 en todas las profundidades, mientras el de ejes
# cae a 0.809 con profundidad 1 y necesita la escalera para recuperarse, lo que confirma que el de ejes no es invariante
# a rotación y el oblicuo sí.

# Si algún modelo no cambia cuando esperabas que cambiara, explica por qué
# Con el logaritmo e^{3x₁} mi árbol tampoco cambia, la explicación es que, como la señal de estos datos depende de un
# solo eje, mi árbol compite en cada nodo el corte oblicuo contra el de eje y elige el de eje, heredando así su
# invariancia a funciones crecientes.