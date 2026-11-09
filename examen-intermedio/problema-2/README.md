# Problema 2 — Árboles que cortan en diagonal
Karol Burrola Torres · 223210891 · a=1, b=9, semilla=223210891.
Datos reales: a mod 4 = 1 → cáncer de mama (WDBC, maligno M = 1, 569 filas, 37% positivas).
Ángulo (b): 10 + 5·9 = 55 grados.

## Reproducir
```
pip install scikit-learn pandas numpy matplotlib
cd problema-2
python arbol_oblicuo_a.py          # assert 6 puntos: w proporcional a (1,-1), 0 errores
python codigo_girar_transformar_b.py  # (b): Sin_girar / Girados 55° / Log + grafica
python codigo_datos_reales.py      # (c): Oblicuo vs Eje vs Bosque vs Bagging + hojas/tiempos
python codigo_conclusiones.py      # (d): Diagonal vs Ejes
```

Los datos reales se descargan solos del UCI.
Si falta la carpeta, crear manualmente `problema-2/resultados` antes de correr.

## (a) Mi árbol (`arbol_oblicuo_a.py`)
Un árbol como el normal, pero en cada nodo también prueba cortar en diagonal con la
dirección de Fisher y se queda con el corte (diagonal o de eje) que deje los grupos más
puros. La prueba con los 6 puntos del examen pasa: encuentra la diagonal y no falla ninguno.

## (b) Girar y transformar (`rotacion.png`)
1000 puntos con clase según x1 > 0.2; luego girados 55° y con log en x1. Se prueba
profundidad 1–8.
- Sin_girar: Los 3 llegan a 0.997, la frontera es fácil para todos.
- Girados: El mío no se mueve (0.997 siempre); el de ejes empieza en 0.81 y necesita
  hacerse profundo para recuperar. O sea: girar sí afecta al de ejes y no al oblicuo.
- Log: Nadie cambia, porque mi árbol termina eligiendo el corte de eje y ese no le
  teme al logaritmo.

## (c) Datos reales (`reales.csv`, métrica AUC por 37% positivas, Prof por CV)
| Modelo | Prof | Acc | AUC | Hojas | Tiempo |
|---|---|---|---|---|---|
| Oblicuo | 2 | 0.960 | 0.970 | 4 | ~1.3s |
| Eje | 2 | 0.928 | 0.954 | 4 | 0.01s |
| Bosque | 5 | 0.956 | 0.990 | — | ~0.6s |
| Bagging_oblicuo | 2 | 0.972 | 0.994 | — | ~6s |

Decisiones: Métrica AUC (el 37% desbalancea la accuracy) + Acc de apoyo; Prof elegida por validación cruzada barriendo 1–8 por modelo (el bagging hereda la del oblicuo). El Oblicuo supera al Eje con las mismas 4 hojas; el bagging de oblicuos gana a todo.

## (d) Cuándo vale la pena (`codigo_conclusiones.py`)
Gana donde la frontera es diagonal o las variables están correlacionadas (Diagonal: Oblicuo 0.994 vs Eje 0.904). Empata donde la frontera ya es de ejes (Ejes: 0.998 vs 0.998) gracias a que compite contra el corte de eje. En mis datos no perdió (0.970 vs 0.954): El cáncer tiene variables correlacionadas tipo diagonal, y este experimento lo pone a prueba.

