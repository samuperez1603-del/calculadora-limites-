import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as pgr


# Configuración de página
st.set_page_config(page_title="Calculadora de límites",
                   layout="centered",
                   page_icon="https://cdn-icons-png.flaticon.com/512/2374/2374370.png")


st.markdown("<h1 style = 'color : white; font-family: roboto ;'>calculadora de limites</h1>",unsafe_allow_html=True)


# Entrada de datos
funcion_st = st.text_input("Ingrese f(x):", "(x**2 - 1)/(x - 1)")
punto = st.number_input("Límite cuando x ->", value=1.0)

# Definición simbólica
x = sp.symbols('x')
f = sp.sympify(funcion_st)
limite = sp.limit(f, x, punto)

st.write(" Resultado:", limite)

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

mask = np.isfinite(Y)

limite_izq=sp.limit(f,x,dir='-')
limite_der=sp.limit(f,x,dir='+')
fig=pgr.Figure()


if (limite_izq != limite_der) or (not limite_izq.is_real) or (not limite_der.is_real):
    # Partimos la gráfica en dos
            mask_left = X < punto
            mask_right = X > punto

            fig.add_trace(pgr.Scatter(
                x=X[mask_left], y=Y[mask_left],
                mode="lines",
                line=dict(width=3, color="blue"),
                name="f(x) izquierda"
            ))
            fig.add_trace(pgr.Scatter(
                x=X[mask_right], y=Y[mask_right],
                mode="lines",
                line=dict(width=3, color="green"),
                name="f(x) derecha"
            ))
else:
            # Una sola curva continua
            fig.add_trace(pgr.Scatter(
                x=X, y=Y,
                mode="lines",
                line=dict(width=3, color="blue"),
                name="f(x)"
            ))
        # Ejes
fig.add_shape(
            type="line", x0=-1e3, x1=1e3, y0=0, y1=0,
            line=dict(color="black", width=2)
        )
fig.add_shape(
            type="line", x0=0, x1=0, y0=-1e3, y1=1e3,
            line=dict(color="black", width=2)
        )

        # Mostrar gráfico
fig.update_layout(
            hovermode="x unified",
            legend=dict(font=dict(size=14, color="black"))
        )

def centrar_cam(fig, x0, y0, zoom=1):
            ancho = 10 / zoom   # cuanto más grande el zoom, más cerca
            alto = 6 / zoom
            fig.update_layout(
                xaxis=dict(range=[x0 - ancho/2, x0 + ancho/2]),
                yaxis=dict(range=[y0 - alto/2, y0 + alto/2])
            )

            return fig
fig = centrar_cam(fig, x0=2, y0=3, zoom=2)
        
fig.update_layout(legend=dict(font=dict(size=14, color="black")))
st.plotly_chart(fig, theme=None, use_container_width=True)

# Función
fig.add_trace(pgr.Scatter(x=X, y=Y, mode="lines", name=f"f(x) = {funcion_st}"))

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