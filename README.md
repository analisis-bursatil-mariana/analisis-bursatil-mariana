# 📈 App de Análisis Bursátil — Mariana Hernández

App interactiva construida en **Python + Streamlit** para el examen final de **Ingeniería Financiera**.  
Permite analizar cualquier acción listada en Yahoo Finance desde una perspectiva técnica, de riesgo, valuación y modelos estadísticos.

---

## 🚀 Funcionalidades principales

### 1. Visión general
- Descarga automática de datos históricos vía `yfinance`.
- Resumen de la empresa (sector, industria y descripción de negocio).
- Precio actual y métricas clave en tarjetas tipo dashboard.

### 2. Análisis técnico
- Gráfico de velas OHLC con:
  - Ventanas rápidas (1M, 3M, 6M, 1Y, 3Y, 5Y, Max) o rango personalizado.
  - Medias móviles configurables (10, 20, 50, 100, 200).
- Indicadores técnicos:
  - **RSI (14)** con bandas 30–70.
  - **MACD (12, 26, 9)**.
- Volumen de operación integrado en el gráfico.

### 3. Riesgo y rendimiento
- Rendimiento total por periodo.
- Volatilidad anualizada.
- Correlación vs. benchmark.
- Beta.
- Ratio de Sharpe.
- VaR histórico al 95%.
- Descarga de métricas a CSV.

### 4. Comparables y compatibilidad
- Identificación automática de empresas comparables.
- Cálculo de correlaciones.
- Mapa de calor visual.

### 5. 💰 Valuación rápida (DCF simplificado)
- Inputs de crecimiento y tasa de descuento.
- Valor intrínseco estimado.
- Diagnóstico automático (subvaluada/sobrevaluada).

### 6. 📉 Proyección ARIMA(1,1,1)
- Pronóstico de precios.
- Intervalos de confianza.
- Gráfico profesional con Plotly.

### 7. 🤖 Regímenes de mercado (KMeans)
- Clusterización de retornos y volatilidad.
- Identificación de estados de mercado.

---

## 🛠️ Tecnologías utilizadas
- Python
- Streamlit
- yfinance
- pandas
- numpy
- plotly
- statsmodels
- scikit-learn

---

## 💻 Cómo ejecutar la app
```bash
git clone https://github.com/analisis-bursatil-mariana/analisis-bursatil-mariana.git
cd analisis-bursatil-mariana
pip install -r requirements.txt
streamlit run claseapp.py
## Hi there 👋

<!--
**analisis-bursatil-mariana/analisis-bursatil-mariana** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
