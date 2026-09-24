# Optimización de carteras de inversión y patrimoniales con pignoración vía aprendizaje automático

Repositorio técnico del Trabajo Fin de Máster (Big Data y Ciencia de Datos, Universidad Internacional de Valencia). Contiene el pipeline reproducible utilizado para modelar el riesgo de carteras de inversión empleadas como colateral en operaciones de financiación con pignoración (crédito Lombard), y para identificar composiciones de cartera que equilibren rentabilidad neta y estabilidad del colateral.

El trabajo no constituye un modelo regulatorio ni un sistema de concesión crediticia real. Es una aproximación académica y exploratoria, construida exclusivamente sobre datos públicos de Yahoo Finance.

## Qué problema aborda

Una operación de pignoración concede liquidez a cambio de una cartera de activos como garantía. El indicador central es el ratio Loan-to-Value (LTV), que compara la deuda con el valor del colateral. Los enfoques tradicionales fijan el LTV de forma estática, sin considerar que el valor de una cartera cambia con el tiempo según su volatilidad, correlación entre activos y drawdown. Cuando el colateral cae por debajo del umbral pactado, se produce un margin call: el cliente debe aportar garantía adicional o la entidad ejecuta la prenda.

Este proyecto modela esa dinámica de forma continua, en vez de tratarla como una regla fija, combinando simulación financiera, aprendizaje automático y optimización de carteras.

## Universo de datos

Cuatro ETFs de acceso público, descargados vía `yfinance`:

| ETF | Proxy de | Moneda de cotización |
|---|---|---|
| XEON.DE | Instrumentos monetarios en euros | EUR |
| SHY | Renta fija de corto plazo (EE. UU.) | USD |
| AGGU.L | Renta fija global agregada | USD |
| SPY | Mercado accionario estadounidense (S&P 500) | USD |

Los tres activos denominados en dólares se convierten a euros con el tipo de cambio EUR/USD histórico de cada fecha (también descargado de Yahoo Finance), de modo que el colateral, la deuda y el coste de financiación queden expresados en una misma unidad monetaria a lo largo de toda la simulación.

La ventana temporal común de análisis va del 21 de noviembre de 2017 a junio de 2026, determinada por la disponibilidad histórica de AGGU.L, el activo más restrictivo.

## Estructura del repositorio

```
optimizacion-carteras-pignoracion-ml/
├── 1_extraccion_datos.ipynb
├── 2_desarrollo_implementacion.ipynb
├── 3_graficos.ipynb
└── data/
    ├── entradas/     (salidas del notebook 1: precios, catálogo de carteras, colateral)
    └── salidas/      (salidas del notebook 2: Monte Carlo, financiación, ML, optimización, tablas)
```

### 1. `1_extraccion_datos.ipynb`: extracción y preparación de datos

- Descarga precios diarios de los 4 ETFs y del tipo de cambio EUR/USD.
- Convierte a euros los activos cotizados en dólares.
- Recorta la serie a la ventana temporal común entre los 4 activos.
- Construye `dataset_mercado`: precios, retornos (1, 21, 63 y 252 días), volatilidad anualizada, drawdown y correlación media rolling.
- Genera `catalogo_carteras`: 286 combinaciones de pesos entre los 4 ETFs (incrementos del 10 %), clasificadas por perfil (conservador, equilibrado, dinámico) según su exposición a renta variable e instrumentos monetarios.
- Construye `dataset_colateral`: evolución diaria del valor de cada cartera (capital inicial de 100.000 €), capitalizando los retornos diarios ponderados por los pesos de cada combinación.

### 2. `2_desarrollo_implementacion.ipynb`: motor del pipeline

Cuatro componentes secuenciales:

**Simulación Monte Carlo multivariante.** Genera 1.000 trayectorias de mercado por cartera a 3 años (756 días), mediante un movimiento browniano geométrico calibrado sobre la matriz de covarianzas histórica de los 4 activos, preservando la dependencia entre ellos. El cálculo se procesa en lotes de carteras para acotar el consumo de memoria.

**Motor de financiación.** Traduce cada trayectoria de colateral en una trayectoria de LTV bajo dos estructuras de deuda:
- Línea de crédito: principal constante durante todo el horizonte.
- Préstamo amortizable: saldo vivo decreciente de forma lineal.

Calcula la probabilidad de margin call (LTV ≥ 80 %), la severidad del evento, el margen de seguridad y la rentabilidad neta del cliente (retorno simulado menos coste de financiación menos TER de la cartera). El coste de financiación se fija en Euribor 3M (3,5 %, aproximación al entorno de tipos 2023-2024) más un spread de 2,5 puntos.

**Aprendizaje automático.**
- *Clasificación (benchmark auxiliar):* Regresión Logística, Random Forest y XGBoost predicen la ocurrencia binaria de margin call. Dada la baja frecuencia del evento (< 0,5 %), el AUC-ROC resulta engañoso; se reporta también PR-AUC, precisión, recall, F1 y Brier score.
- *Regresión (modelo principal):* XGBoost Regressor, Gradient Boosting Regressor y Random Forest Regressor predicen el percentil 95 del LTV máximo simulado, una magnitud continua más informativa que la etiqueta binaria. La partición de entrenamiento/prueba se hace por identificador de cartera para evitar fuga de información. Se calcula importancia de variables por permutación sobre el mejor modelo.

**Optimización de carteras.** Identifica, entre las 286 carteras y ambas estructuras de financiación, aquellas que maximizan la rentabilidad neta media manteniendo la probabilidad de margin call por debajo de un umbral tolerable (5 %).

Esta misma notebook exporta también las tablas agregadas (`tabla_*.parquet`) que alimentan la memoria del TFM y el notebook de gráficos, y ejecuta una validación final de reproducibilidad del pipeline completo.

### 3. `3_graficos.ipynb`: visualizaciones

Genera las ilustraciones de la memoria a partir de las tablas agregadas: distribución de carteras por perfil, riesgo histórico, resultados de la simulación Monte Carlo, comparación de estructuras de financiación, desempeño de los modelos de clasificación y regresión, importancia de variables, frontera riesgo-retorno y validaciones del pipeline.

## Dependencias

```
pandas
numpy
yfinance
scikit-learn
xgboost
matplotlib
joblib
```

## Limitaciones conocidas

- Los ETFs son proxies de clases de activo, no fondos bancarios reales ni carteras privadas completas.
- El dataset supervisado de margin call proviene de simulación, no de eventos reales registrados.
- El coste de financiación se mantiene constante durante toda la simulación (no se modela sensibilidad a variaciones del Euribor).
- El catálogo de carteras es discreto (286 combinaciones con incrementos del 10 %), no un espacio continuo de pesos.
- Los resultados son válidos bajo la ventana histórica y los supuestos declarados; no constituyen una recomendación de inversión.

## Referencias principales

- Anderson, R. W., y Jõeveer, K. (2014). *The economics of collateral*. SRC/FMG Discussion Paper, LSE.
- Kabtoul, A. (2020). *Determining the loan-to-value of structured products for security backed lending*. SSRN Working Paper 3612950.
- López de Prado, M. (2016). *Building diversified portfolios that outperform out-of-sample*. The Journal of Portfolio Management, 42(4), 59–69.
- Saito, T., y Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets*. PLOS ONE, 10(3), e0118432.
- Zhang, R., Zhang, J., y Xu, S. (2015). *Determining pledged loan-to-value ratio: An option pricing perspective*. Financial Innovation, 1, 16.

La lista completa de referencias, con su aporte específico a cada sección del trabajo, está en el Anexo I de la memoria del TFM.

## Autor

Wilmer Ruiz Camacho, Máster en Big Data y Ciencia de Datos, Universidad Internacional de Valencia (VIU). 

## Director

Enrique De Miguel Ambite, Universidad Internacional de Valencia (VIU). 
