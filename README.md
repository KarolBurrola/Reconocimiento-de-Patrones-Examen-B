# Examen Intermedio B — Reconocimiento de Patrones 

Nombre: Burrola Torres Karol · Expediente: 223210891 
Profesor: Julio Waissman Vilanova · Semestre 2026-2.

## Estructura
```
examen-intermedio/
|-- README.md              # Este archivo: datos, parametros y como reproducir todo
|-- problema-1/            # Practica problema 1 (Bank Marketing UCI)
|-- problema-2/            # Practica problema 2 (Arbol oblicuo + Cancer mama UCI)
```

## Parametros personales
- Semilla : 223210891
- a = 1 → Problema 1: k = 11, RandomForest (b mod 3 = 0); Problema 2: datos de cáncer de mama.
- b = 9 → Problema 1: n = 1000, 65 atributos en ruido; Problema 2: giro de 55°.

## Reproducir todo
```
pip install scikit-learn pandas numpy matplotlib

cd examen-intermedio/problema-1
python codigoejemplo.py            
python codigocorrecciones.py        
python codigohonesto.py            
python codigoruido.py              

cd ../problema-2
python arbol_oblicuo_a.py          
python codigo_girar_transformar_b.py  
python codigo_datos_reales.py       
python codigo_conclusiones.py       
```

