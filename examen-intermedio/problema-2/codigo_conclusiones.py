import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from arbol_oblicuo_a import ArbolOblicuo

# Cuando vale la pena: diagonal (favorable) vs ejes (desfavorable)
# Si el oblicuo pierde en datos reales, la explicacion es que la
# frontera real es de ejes y el oblicuo mete varianza de gratis.

SEED = 223210891
rng = np.random.default_rng(SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

X = rng.uniform(0, 1, (500, 2))
y_diag = (X[:, 1] > X[:, 0]).astype(int)
y_ejes = (X[:, 0] > 0.5).astype(int)

print(f"{'Datos':<10} | {'Oblicuo':<7} | {'Eje'}")
print("-" * 28)
for nombre, yy in [("Diagonal", y_diag), ("Ejes", y_ejes)]:
    a = cross_val_score(ArbolOblicuo(max_prof=3), X, yy, cv=cv).mean()
    b = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=SEED), X, yy, cv=cv).mean()
    print(f"{nombre:<10} | {a:<7.3f} | {b:.3f}")
