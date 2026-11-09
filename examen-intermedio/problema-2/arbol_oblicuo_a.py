import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def gini(etiquetas):
    if len(etiquetas) == 0:
        return 0.0
    proporcion_unos = np.mean(etiquetas)
    return 1.0 - proporcion_unos ** 2 - (1.0 - proporcion_unos) ** 2


def mejor_umbral(valores_proyectados, etiquetas):
    indice_orden = np.argsort(valores_proyectados)
    valores_proyectados = valores_proyectados[indice_orden]
    etiquetas = etiquetas[indice_orden]
    mejor_gini_umbral = (1.0, 0.0)
    for indice_corte in range(1, len(valores_proyectados)):
        if valores_proyectados[indice_corte] == valores_proyectados[indice_corte - 1]:
            continue
        umbral_candidato = (valores_proyectados[indice_corte] + valores_proyectados[indice_corte - 1]) / 2.0
        gini_ponderado = (indice_corte * gini(etiquetas[:indice_corte])
                          + (len(etiquetas) - indice_corte) * gini(etiquetas[indice_corte:])) / len(etiquetas)
        if gini_ponderado < mejor_gini_umbral[0]:
            mejor_gini_umbral = (gini_ponderado, umbral_candidato)
    return mejor_gini_umbral


def direccion_fisher(X, y, lam=1e-3):
    media_clase0 = X[y == 0].mean(axis=0)
    media_clase1 = X[y == 1].mean(axis=0)
    centrados_clase0 = X[y == 0] - media_clase0
    centrados_clase1 = X[y == 1] - media_clase1
    dispersion_total = centrados_clase0.T @ centrados_clase0 + centrados_clase1.T @ centrados_clase1
    dispersion_total = dispersion_total + lam * np.eye(X.shape[1])
    return np.linalg.solve(dispersion_total, media_clase1 - media_clase0)


class ArbolOblicuo(BaseEstimator, ClassifierMixin):
    def __init__(self, max_prof=5, min_muestras=10, lam=1e-3, competir=True):
        self.max_prof = max_prof
        self.min_muestras = min_muestras
        self.lam = lam
        self.competir = competir

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.n_clases_ = 2
        self.arbol_ = self.nodo(X, y, 0)
        return self

    def nodo(self, X, y, profundidad):
        if len(y) < self.min_muestras or np.all(y == y[0]) or profundidad >= self.max_prof:
            return {"es_hoja": True, "probabilidad": float(np.mean(y))}
        n_datos, n_variables = X.shape
        direccion = direccion_fisher(X, y, self.lam)
        proyeccion = X @ direccion
        gini_oblicuo, umbral_oblicuo = mejor_umbral(proyeccion, y)
        gini_eje, variable_eje, umbral_eje = 1.0, 0, 0.0
        for variable in range(n_variables):
            gini_candidato, umbral_candidato = mejor_umbral(X[:, variable], y)
            if gini_candidato < gini_eje:
                gini_eje, variable_eje, umbral_eje = gini_candidato, variable, umbral_candidato
        if self.competir and gini_eje <= gini_oblicuo:
            variable_ganadora = variable_eje
            mascara_izquierda = X[:, variable_ganadora] <= umbral_eje
        else:
            variable_ganadora = None
            mascara_izquierda = proyeccion <= umbral_oblicuo
        if mascara_izquierda.sum() == 0 or mascara_izquierda.sum() == n_datos:
            return {"es_hoja": True, "probabilidad": float(np.mean(y))}
        nodo_actual = {"es_hoja": False, "direccion": direccion, "umbral": float(umbral_oblicuo),
                       "variable": variable_ganadora, "umbral_eje": float(umbral_eje)}
        nodo_actual["izquierda"] = self.nodo(X[mascara_izquierda], y[mascara_izquierda], profundidad + 1)
        nodo_actual["derecha"] = self.nodo(X[~mascara_izquierda], y[~mascara_izquierda], profundidad + 1)
        return nodo_actual

    def prob(self, X):
        X = np.asarray(X, dtype=float)
        probabilidades = np.zeros(len(X))
        for indice, punto in enumerate(X):
            nodo_actual = self.arbol_
            while not nodo_actual["es_hoja"]:
                if nodo_actual["variable"] is None:
                    va_izquierda = (punto @ nodo_actual["direccion"]) <= nodo_actual["umbral"]
                else:
                    va_izquierda = punto[nodo_actual["variable"]] <= nodo_actual["umbral_eje"]
                nodo_actual = nodo_actual["izquierda"] if va_izquierda else nodo_actual["derecha"]
            probabilidades[indice] = nodo_actual["probabilidad"]
        return probabilidades

    def predict_proba(self, X):
        probabilidades = self.prob(X)
        return np.c_[1.0 - probabilidades, probabilidades]

    def decision_function(self, X):
        return self.prob(X)

    def predict(self, X):
        return (self.prob(X) >= 0.5).astype(int)

    def n_hojas(self):
        def contar_hojas(nodo_actual):
            if nodo_actual["es_hoja"]:
                return 1
            return contar_hojas(nodo_actual["izquierda"]) + contar_hojas(nodo_actual["derecha"])
        return contar_hojas(self.arbol_)


if __name__ == "__main__":
    X = np.array([[0, 1], [1, 3], [3, 4], [1, 0], [3, 1], [4, 3]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    direccion = direccion_fisher(X, y)
    assert abs(direccion[0] + direccion[1]) < 1e-8 and direccion[0] > 0, direccion
    mi_arbol = ArbolOblicuo(max_prof=3, min_muestras=2, competir=False).fit(X, y)
    assert (mi_arbol.predict(X) == y).all()
    print("assert ok: direccion =", direccion, "hojas =", mi_arbol.n_hojas())