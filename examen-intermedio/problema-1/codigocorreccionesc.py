import io, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
externo = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
interno = zipfile.ZipFile(io.BytesIO(externo.read("bank.zip")))
datos = pd.read_csv(interno.open("bank-full.csv"), sep=";")

y = (datos.pop("y") == "yes").astype(int)
X = pd.get_dummies(datos, drop_first=True).astype(float)

filas = []
prof_fija = 8  # Para comparar variantes sin mezclar con la eleccion del mejor

print("")
print("Tabla (b): problemas en el codigo de la consultora")
print(f"{'#':<3} | {'Problema':<42} | {'Tipo':<26} | {'¿Por qué infla?'}")
print(f"{'1':<3} | {'Duplicar positivos ANTES del split':<42} | {'Fuga por muestra':<26} | 99.99% de positivos de prueba tiene gemelo en train")
print(f"{'2':<3} | {'StandardScaler con todos los datos':<42} | {'Fuga de preproceso':<26} | La prueba aporta media/varianza al train")
print(f"{'3':<3} | {'SelectKBest(k=11) con todos los datos':<42} | {'Mala practica de validacion':<26} | Se elige viendo etiquetas de prueba")
print(f"{'4':<3} | {'Elegir profundidad con la prueba':<42} | {'Reuso de prueba':<26} | El ganador gana en parte por suerte")
print(f"{'5':<3} | {'Usar duration':<42} | {'Fuga de informacion':<26} | Solo se conoce al colgar, una pre-llamada no existe")
print(f"{'6':<3} | {'Split al azar ignorando cronologia':<42} | {'Mala validacion temporal':<26} | Mezcla 2008-2010")
print(f"{'7':<3} | {'Accuracy sin intervalo en desbalance':<42} | {'Metrica pobre':<26} | No mide priorizar a quien llamar")
print("")

# Código base de la consultora
pos = y == 1
X_dup = pd.concat([X] + [X[pos]] * 6, ignore_index=True)
y_dup = pd.concat([y] + [y[pos]] * 6, ignore_index=True)
X_tmp = StandardScaler().fit_transform(X_dup)
X_tmp = SelectKBest(f_classif, k=11).fit_transform(X_tmp, y_dup)
X_ent, X_pru, y_ent, y_pru = train_test_split(X_tmp, y_dup, test_size=0.2, random_state=223210891)
mejor = None
for prof in [4, 8, 16, None]:
    modelo = RandomForestClassifier(n_estimators=200, max_depth=prof,
                                    random_state=223210891, n_jobs=-1)
    modelo.fit(X_ent, y_ent)
    auc = roc_auc_score(y_pru, modelo.predict_proba(X_pru)[:, 1])
    if mejor is None or auc > mejor[0]:
        mejor = (auc, prof, accuracy_score(y_pru, modelo.predict(X_pru)))
filas.append(("Codigo_consultora", mejor[2], mejor[0]))
print(f"Codigo_consultora acc={mejor[2]:.4f} auc={mejor[0]:.4f}")

# 1) Sin duplicar: Se parte primero y no se copia nada
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=223210891)
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=11),
                     RandomForestClassifier(n_estimators=200, max_depth=prof_fija,
                                            random_state=223210891, n_jobs=-1))
pipe.fit(X_ent, y_ent)
acc = accuracy_score(y_pru, pipe.predict(X_pru))
auc = roc_auc_score(y_pru, pipe.predict_proba(X_pru)[:, 1])
filas.append(("1_Sin_duplicar", acc, auc))
print(f"1_Sin_duplicar acc={acc:.4f} auc={auc:.4f}")

# 2 y 3) Preproceso dentro del train, pero duplicando solo el train
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=223210891)
pos = y_ent == 1
X_ent = pd.concat([X_ent] + [X_ent[pos]] * 6, ignore_index=True)
y_ent = pd.concat([y_ent] + [y_ent[pos]] * 6, ignore_index=True)
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=11),
                     RandomForestClassifier(n_estimators=200, max_depth=prof_fija,
                                            random_state=223210891, n_jobs=-1))
pipe.fit(X_ent, y_ent)
acc = accuracy_score(y_pru, pipe.predict(X_pru))
auc = roc_auc_score(y_pru, pipe.predict_proba(X_pru)[:, 1])
filas.append(("2_3_Preproceso_train_dup", acc, auc))
print(f"2_3_Preproceso_train_dup acc={acc:.4f} auc={auc:.4f}")

# 4) Elegir prof con validacion interna, no con la prueba
X_ent, X_pru, y_ent, y_pru = train_test_split(X_tmp, y_dup, test_size=0.2, random_state=223210891)
X_sub, X_val, y_sub, y_val = train_test_split(X_ent, y_ent, test_size=0.25, random_state=223210891)
mejor = None
for prof in [4, 8, 16, None]:
    modelo = RandomForestClassifier(n_estimators=200, max_depth=prof,
                                    random_state=223210891, n_jobs=-1)
    modelo.fit(X_sub, y_sub)
    auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])
    if mejor is None or auc > mejor[0]:
        mejor = (auc, prof)
modelo = RandomForestClassifier(n_estimators=200, max_depth=mejor[1],
                                random_state=223210891, n_jobs=-1)
modelo.fit(X_ent, y_ent)
acc = accuracy_score(y_pru, modelo.predict(X_pru))
auc = roc_auc_score(y_pru, modelo.predict_proba(X_pru)[:, 1])
filas.append((f"4_Validacion_Interna(prof={mejor[1]})", acc, auc))
print(f"4_Validacion_Interna(prof={mejor[1]}) acc={acc:.4f} auc={auc:.4f}")

# 5) Sin duration (se quita antes)
X_sin = X.drop(columns=[c for c in X.columns if c.startswith("duration")])
X_ent, X_pru, y_ent, y_pru = train_test_split(X_sin, y, test_size=0.2, random_state=223210891)
pos = y_ent == 1
X_ent = pd.concat([X_ent] + [X_ent[pos]] * 6, ignore_index=True)
y_ent = pd.concat([y_ent] + [y_ent[pos]] * 6, ignore_index=True)
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=11),
                     RandomForestClassifier(n_estimators=200, max_depth=prof_fija,
                                            random_state=223210891, n_jobs=-1))
pipe.fit(X_ent, y_ent)
acc = accuracy_score(y_pru, pipe.predict(X_pru))
auc = roc_auc_score(y_pru, pipe.predict_proba(X_pru)[:, 1])
filas.append(("5_Sin_duration", acc, auc))
print(f"5_Sin_duration acc={acc:.4f} auc={auc:.4f}")

# 6) Split temporal: Ultimo 20% cronologico como prueba
corte = int(len(X) * 0.8)
X_ent, X_pru, y_ent, y_pru = X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]
pos = y_ent == 1
X_ent = pd.concat([X_ent] + [X_ent[pos]] * 6, ignore_index=True)
y_ent = pd.concat([y_ent] + [y_ent[pos]] * 6, ignore_index=True)
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=11),
                     RandomForestClassifier(n_estimators=200, max_depth=prof_fija,
                                            random_state=223210891, n_jobs=-1))
pipe.fit(X_ent, y_ent)
acc = accuracy_score(y_pru, pipe.predict(X_pru))
auc = roc_auc_score(y_pru, pipe.predict_proba(X_pru)[:, 1])
filas.append(("6_Temporal_con_dup", acc, auc))
print(f"6_Temporal_con_dup acc={acc:.4f} auc={auc:.4f}")

# 7) Juntar: Temporal + sin duration + duplica solo el train
X_sin = X.drop(columns=[c for c in X.columns if c.startswith("duration")])
X_ent, X_pru, y_ent, y_pru = X_sin.iloc[:corte], X_sin.iloc[corte:], y.iloc[:corte], y.iloc[corte:]
pos = y_ent == 1
X_ent = pd.concat([X_ent] + [X_ent[pos]] * 6, ignore_index=True)
y_ent = pd.concat([y_ent] + [y_ent[pos]] * 6, ignore_index=True)
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=11),
                     RandomForestClassifier(n_estimators=200, max_depth=prof_fija,
                                            random_state=223210891, n_jobs=-1))
pipe.fit(X_ent, y_ent)
acc = accuracy_score(y_pru, pipe.predict(X_pru))
auc = roc_auc_score(y_pru, pipe.predict_proba(X_pru)[:, 1])
filas.append(("7_juntar_temporal", acc, auc))
print(f"7_juntar_temporal acc={acc:.4f} auc={auc:.4f}")

# 7b) Juntar: Split al azar
X_ent, X_pru, y_ent, y_pru = train_test_split(X_sin, y, test_size=0.2, random_state=223210891)
pos = y_ent == 1
X_ent = pd.concat([X_ent] + [X_ent[pos]] * 6, ignore_index=True)
y_ent = pd.concat([y_ent] + [y_ent[pos]] * 6, ignore_index=True)
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=11),
                     RandomForestClassifier(n_estimators=200, max_depth=prof_fija,
                                            random_state=223210891, n_jobs=-1))
pipe.fit(X_ent, y_ent)
acc = accuracy_score(y_pru, pipe.predict(X_pru))
auc = roc_auc_score(y_pru, pipe.predict_proba(X_pru)[:, 1])
filas.append(("7b_juntar_azar", acc, auc))
print(f"7b_juntar_azar acc={acc:.4f} auc={auc:.4f}")

pd.DataFrame(filas, columns=["variante", "acc", "auc"]).to_csv("resultados/tablacomparativa.csv", index=False)


# -- PREGUNTAS --

# ¿Cuál pesa más?
# Lo que más pesa es quitar duration, el AUC cae de 0.9698 a 0.7520, esto confirma que duration era la fuga principal,
# pues solo se conoce al finalizar la llamada y no está disponible al momento de priorizar.


# ¿Hay alguno que casi no cambie el resultado? ¿Por qué?
# La selección con validación interna da idéntico a la base (0.9126 y 0.9698, ganando prof=None de nuevo), lo cual
# coincide con mi inciso (b), donde el sesgo de elegir 4 modelos en una prueba de aproximadamente 15 mil datos es de
# solo medio punto. Asimismo, ajustar scaler y select dentro del train casi no mueve el AUC (0.9057 a 0.9043), lo
# esperado con 77 mil renglones, donde la información que aporta la prueba al preproceso es malísimo.


# Compara el efecto de las copias con tu fórmula del inciso (a).
# Mi fórmula predice 0.48+0.52·p. Tomando p ≈ 0.81, que es el desempeño sin memorización, se predice aproximadamente
# 0.90, muy cercano al 0.9126 medido en la base. En AUC el efecto es de 0.9698 a 0.9057, cerca de 6 puntos, pues el AUC
# sí refleja el ordenamiento de los positivos clonados.