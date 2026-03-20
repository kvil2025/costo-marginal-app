# Dashboard de Análisis de Costo Marginal (CMARG)

Este proyecto implementa un dashboard interactivo para el análisis y la visualización de datos de Costo Marginal del Coordinador Eléctrico Nacional de Chile.

## Características

- **Visualización Interactiva**: Gráficos dinámicos construidos con Plotly y Dash.
- **Filtros Avanzados**: Permite filtrar los datos por rango de fechas, centrales eléctricas y agrupación temporal (hora, día, semana, mes).
- **Análisis Comparativo**: Compara la distribución del costo marginal entre diferentes centrales a través de box plots.
- **Manejo Eficiente de Datos**: La aplicación está optimizada para procesar grandes volúmenes de datos en el servidor, evitando problemas de memoria en el navegador.

## Cómo Empezar

Sigue estos pasos para configurar y ejecutar el proyecto en tu entorno local.

### Prerrequisitos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. Clona o descarga este repositorio en tu máquina local.

2. Navega al directorio raíz del proyecto:
   ```bash
   cd CMARG
   ```

3. Instala las dependencias necesarias ejecutando:
   ```bash
   pip install -r requirements.txt
   ```

### Estructura de Datos

Asegúrate de que tus archivos de datos (`.tsv`) se encuentren en el directorio `data/raw/`.

### Ejecución

1. Para iniciar el dashboard, ejecuta el siguiente comando desde el directorio raíz del proyecto:
   ```bash
   python app.py
   ```

2. Abre tu navegador web y ve a la siguiente dirección:
   [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

## Estructura del Proyecto

```
CMARG/
├── data/
│   └── raw/         # Archivos de datos TSV originales
├── src/
│   └── data/
│       └── loader.py  # Módulo para cargar y procesar los datos
├── app.py             # Script principal de la aplicación Dash
├── requirements.txt   # Lista de dependencias de Python
└── README.md          # Este archivo
```
