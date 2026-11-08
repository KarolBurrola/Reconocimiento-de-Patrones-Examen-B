import numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# Mis parametros: n = 50 datos, 5000 ruido, 20 + 5*b = 65 atributos (b = 9)
# Clasificador: LogisticRegression

rng = np.random.default_rng(223210891)
antes, dentro = [], []
REPS = 60
for r in range(REPS):
    X = rng.standard_normal((50, 5000))
    y = rng.integers(0, 2, 50)
    if len(np.unique(y)) < 2:
        y[0], y[1] = 0, 1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=223210891 + r)
    X_sel = SelectKBest(f_classif, k=65).fit_transform(X, y)
    modelo = LogisticRegression(max_iter=2000)
    antes.append(cross_val_score(modelo, X_sel, y, cv=cv).mean())
    modelo = make_pipeline(SelectKBest(f_classif, k=65),
                           LogisticRegression(max_iter=2000))
    dentro.append(cross_val_score(modelo, X, y, cv=cv).mean())
    print(f"Rep {r+1}/{REPS} Antes={antes[-1]:.3f} Dentro={dentro[-1]:.3f}")

print(f"Antes: Media={np.mean(antes):.3f} Dentro: Media={np.mean(dentro):.3f}")
pd.DataFrame({"Antes": antes, "Dentro": dentro}).to_csv("resultados/ruido.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure()
plt.hist(antes, bins=15, alpha=0.6, label=f"Seleccion ANTES (media {np.mean(antes):.2f})")
plt.hist(dentro, bins=15, alpha=0.6, label=f"Seleccion DENTRO (media {np.mean(dentro):.2f})")
plt.axvline(0.5, color="black", linestyle="--", label="Azar (0.50)")
plt.xlabel("Exactitud estimada (CV-5)")
plt.ylabel("Repeticiones")
plt.legend()
plt.tight_layout()
plt.savefig("resultados/ruido.png", dpi=150)
print("grafica: resultados/ruido.png")
