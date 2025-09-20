import streamlit as str
import sympy as sp
import numpy as np
import plotly.graph_objects as pgr


# Configuración de página
str.set_page_config(page_title="Calculadora de límites",
                   layout="centered",
                   page_icon="https://cdn-icons-png.flaticon.com/512/2374/2374370.png")


str.markdown("<h1 style = 'color : white; font-family: roboto ;'>calculadora de limites</h1>",unsafe_allow_html=True)


# Entrada de datos
funcion_str = str.text_input("Ingrese f(x):", "(x**2 - 1)/(x - 1)")
punto = str.number_input("Límite cuando x ->", value=1.0)

# Definición simbólica
x = sp.symbols('x')
f = sp.sympify(funcion_str)
limite = sp.limit(f, x, punto)

str.write(" Resultado:", limite)

# --- Gráfico interactivo con Plotly ---
f_lamb = sp.lambdify(x, f, "numpy")

# Rango de valores (evitamos división por cero en caso de singularidades)
X = np.linspace(punto-5, punto+5, 400)
Y = f_lamb(X)
fig = pgr.Figure()

# Rango para los ejes (más largos)
x_min, x_max = -20, 20   # Eje X más largo
y_min, y_max = -200, 200  # Eje Y más largo
# Eje X (horizontal negro)
fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0,
              line=dict(color="black", width=3))

# Eje Y (vertical negro)
fig.add_shape(type="line", x0=0, x1=0, y0=y_min, y1=y_max,
              line=dict(color="black", width=3))


# Función
fig.add_trace(pgr.Scatter(x=X, y=Y, mode="lines", name=f"f(x) = {funcion_str}"))

# Línea roja en el punto (con largo más grande)
fig.add_shape(type="line", x0=punto, x1=punto, y0=y_min, y1=y_max,
              line=dict(color="red", dash="dash", width=3))
# Estilo
fig.update_layout(
    title="Gráfica interactiva de f(x)",
    xaxis=dict(title="Eje X", zeroline=False),
    yaxis=dict(title="Eje Y", zeroline=False),
)

str.plotly_chart(fig)