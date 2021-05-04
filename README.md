# Análisis y procesamiento de imágenes - Proyecto

---

A continuación se incluye un log con las características principales de cada entrega.

## Entrega 1

El archivo `explore.py` guía una exploración de las imágenes y anotaciones, selecionando aleatoriamente 2 o más imágenes de la base de datos (link [aquí](https://www.kaggle.com/andrewmvd/face-mask-detection)) con las anotaciones respectivas, y finalmente, un histograma de las frecuencias de las clases presentes en la base de datos. 

## Entrega 2

Corrección en las gráficas del typo donde se conjugaba incorrectamente *worn* como *weared*. Además, se corrigió la imagen de muestra de la base de datos para mostrar el área de detección con color dependiento de la clase. Se agregó código que permitiría escribir el nombre de las clases sobre el rectángulo del área de detección; sin embargo, se optó por una leyenda codificada por colores para evitar problemas de claridad (muchas anotaciones sobrelapadas) cuando existen múltiples anotaciones en una imagen (para activar, modificar `write_annotations` de `False` a `True`). 

## Entrega 3

En el archivo `model_1` se encuentra el primer intento a la solución al problema. Se usa el modelo pre-entrenado Haar Cascade de OpenCV (2001) para caras frontales para detectar las caras en la imagen. Hecho esto, se entrenó un algoritmo VGG19 con una base de datos aparte con aproximadamente 12 mil caras (link [aquí](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)) y se determinaron los tamaños de cara esperados (re-escalamientos), el formato de color de la imagen (escala de grises), y la salida. Hecho esto, la tarea consiste en detectar las caras, re-escalar, aplicar el modelo y determinar si el individuo usó o no el tapabocas.
