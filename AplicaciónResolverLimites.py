import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as pgr
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

st.set_page_config(page_title="Calculadora de límites")

st.markdown("<h1 style='font-family:Arial'> Calculadora de Límites</h1>", unsafe_allow_html=True)

transformaciones = (standard_transformations + (implicit_multiplication_application,))

# Ingreso de función y punto
funcion_str = st.text_input(
    "Escribe tu función f(x):",
    placeholder="(x**2-1)/(x-1)",
    key="funcion_input"
)
punto = st.number_input("Límite cuando x ->", value=1.0)

if st.button("Calcular"):
    x = sp.symbols('x')
    try:
        # Parsear expresión
        f = parse_expr(funcion_str, transformations=transformaciones)

        # Calcular límite
        limite = sp.limit(f, x, punto)
        st.success(f"El límite es: {limite}")

        # Intervalo para graficar
        amplitud = 4000
        N = 1000000
        X = np.linspace(punto - amplitud, punto + amplitud, N)

        # Convertir función a numpy
        if f.is_constant():
            f_const = float(f)  
            f_lamb = lambda X: np.full_like(X, f_const, dtype=float)
        else:
            f_lamb = sp.lambdify(x, f, "numpy")

        Y = f_lamb(X)
        Y[np.abs(Y) > 1e3] = np.nan  # evitar valores enormes

        # Calcular límites laterales
        limite_izq = sp.limit(f, x, punto, dir='-')
        limite_der = sp.limit(f, x, punto, dir='+')


        if np.any(np.isfinite(Y)):
            y_min, y_max = np.nanmin(Y), np.nanmax(Y)
            if y_min == y_max:
                y_min -= 1
                y_max += 1
        else:
            y_min, y_max = -10, 10

        # Crear figura
        fig = pgr.Figure()
        fig.update_layout(
            title=dict(text="Gráfica interactiva de f(x)",
                    font=dict(size=22, color="white"),
                    x=0.5),
            xaxis=dict(
                title=dict(text="Eje X", font=dict(size=16, color="black")),
                showgrid=True, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black",
                zeroline=False,
                tickfont=dict(size=14, color="black")
            ),
            yaxis=dict(
                title=dict(text="Eje Y", font=dict(size=16, color="black")),
                showgrid=True, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black",
                zeroline=False,
                tickfont=dict(size=14, color="black")
            ),
            plot_bgcolor="white",
            paper_bgcolor="lightyellow"
        )
        # Punto del límite
        fig = pgr.Figure()
        fig.update_layout(
            title=dict(text="Gráfica interactiva de f(x)",
                    font=dict(size=22, color="darkblue"),
                    x=0.5),
            xaxis=dict(
                title=dict(text="Eje X", font=dict(size=16, color="black")),
                showgrid=True, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black",
                zeroline=False,
                tickfont=dict(size=14, color="black")
            ),
            yaxis=dict(
                title=dict(text="Eje Y", font=dict(size=16, color="black")),
                showgrid=True, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black",
                zeroline=False,
                tickfont=dict(size=14, color="black")
            ),
            plot_bgcolor="white",
            paper_bgcolor="lightyellow"
        )

        # Punto del límite
        if limite.is_real and limite.is_finite:
            limite_val = float(limite)
            fig.add_trace(pgr.Scatter(
                x=[punto], y=[limite_val],
                mode="markers+text",
                marker=dict(size=12, color="red", symbol="circle"),
                text=[f"({punto}, {limite_val:.2f})"],
                textposition="top right",
                textfont=dict(size=16, color="black", family="Arial"),
                name="Punto del límite"
            ))

        limite_izq = sp.limit(f, x, punto, dir='-')
        limite_der = sp.limit(f, x, punto, dir='+')

        mask = np.isfinite(Y)

        if (limite_izq != limite_der) or (not limite_izq.is_real) or (not limite_der.is_real):
            # Partimos la gráfica en dos
            mask_left = X < punto
            mask_right = X > punto

            fig.add_trace(pgr.Scatter(
                x=X[mask_left], y=Y[mask_left],
                mode="lines",
                line=dict(width=3, color="red"),
                name="f(x) izquierda"
            ))
            fig.add_trace(pgr.Scatter(
                x=X[mask_right], y=Y[mask_right],
                mode="lines",
                line=dict(width=3, color="blue"),
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

    except Exception as e:
        st.error(f"Error en la función ingresada: {e}")