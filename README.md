# Análisis y procesamiento de imágenes - Proyecto

---

A continuación se incluye un log con las características principales de cada entrega.

## Entrega 1

El archivo `explore.py` guía una exploración de las imágenes y anotaciones, selecionando aleatoriamente 2 o más imágenes de la base de datos (link [aquí](https://www.kaggle.com/andrewmvd/face-mask-detection)) con las anotaciones respectivas, y finalmente, un histograma de las frecuencias de las clases presentes en la base de datos. 

## Entrega 2

Corrección en las gráficas del typo donde se conjugaba incorrectamente *worn* como *weared*. Además, se corrigió la imagen de muestra de la base de datos para mostrar el área de detección con color dependiento de la clase. Se agregó código que permitiría escribir el nombre de las clases sobre el rectángulo del área de detección; sin embargo, se optó por una leyenda codificada por colores para evitar problemas de claridad (muchas anotaciones sobrelapadas) cuando existen múltiples anotaciones en una imagen (para activar, modificar `write_annotations` de `False` a `True`). 