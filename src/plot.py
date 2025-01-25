import datetime
import yfinance as yf
import pandas as pd
import matplotlib as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import json
from .finance import int_to_date,date_to_int


colors = [
    "rgba(207,255,36,1)",  # Yellow-Green
    "rgba(0,255,255,1)",    # Cyan
    "rgba(255,165,0,1)",    # Orange
    "rgba(255,0,0,1)",      # Red
    "rgba(0,0,255,1)",      # Blue
    "rgba(138,43,226,1)",   # BlueViolet
    "rgba(255,20,147,1)",   # DeepPink
    "rgba(0,128,0,1)",      # Green
    "rgba(255,105,180,1)",  # HotPink
    "rgba(75,0,130,1)",     # Indigo
]

def get_plotly_histogram(title: str, xaxis_title: str, yaxis_title: str):
    fig = go.Figure()
    fig.update_layout(
        title={
            "text": title,
            "font": {"color": "white"},  # Title text color
        },
        xaxis={
            "title": {"text": xaxis_title, "font": {"color": "rgba(207,255,36,1)"}},  # X-axis title color
            "tickfont": {"color": "rgba(207,255,36,1)"},  # X-axis tick font color
        },
        yaxis={
            "title": {"text": yaxis_title, "font": {"color": "rgba(207,255,36,1)"}},  # Y-axis title color
            "tickfont": {"color": "rgba(207,255,36,1)"},  # Y-axis tick font color
        },
        plot_bgcolor="black",  # Plot background color
        paper_bgcolor="black",  # Overall figure background color
        font={"color": "rgba(207,255,36,1)"},  # Default text color
    )
    return fig

def get_plot_histogram(df_multi_actifs: pd.DataFrame):
    figure = get_plotly_histogram("Distribution de l'investissement par actif", "Investissement (USD)", "Fréquence")
    
    # Iterate over each asset (key) and assign a different color
    for idx, (key, df_actif) in enumerate(df_multi_actifs.T.groupby(level=0)):
        # Assign color from the palette
        color = colors[idx % len(colors)]  # Ensures it wraps around the color palette if more assets are present
        figure.add_trace(go.Histogram(
            x=df_actif.droplevel(0).T['investissement_cumule'], 
            name=key,
            nbinsx=20,  # Number of bins, adjust as needed
            marker=dict(color=color)  # Assign color for each asset
        ))
    
    return figure


def get_plotly_figure(title: str, xaxis_title: str, yaxis_title: str):
    fig = go.Figure()
    fig.update_layout(
        title={
            "text": title,
            "font": {"color": "white"},  # Text color: CFFF24
        },
        xaxis={
            "title": {"text": xaxis_title, "font": {"color": "rgba(207,255,36,1)"}},
            "tickfont": {"color": "rgba(207,255,36,1)"},
        },
        yaxis={
            "title": {"text": yaxis_title, "font": {"color": "rgba(207,255,36,1)"}},
            "tickfont": {"color": "rgba(207,255,36,1)"},
        },
        plot_bgcolor="black",  # Plot background color
        paper_bgcolor="black",  # Overall figure background color
        font={"color": "rgba(207,255,36,1)"},  # Default text color
    )
    return fig


def add_trace_plotly(fig: go.Figure, trace_data: pd.Series, label: str, mode: str, legendgroup=None ,visible=True):
    fig.add_trace(
        go.Scatter(
            x=trace_data.index,
            y=trace_data,
            mode=mode,
            name=label,
            showlegend=True,
            legendgroup=legendgroup,
            visible=visible
        )
    )
    return fig

def convert_plotly_to_json(figure):
    json_plot = plotly.io.to_json(figure)
    return json_plot

def get_plot_adj_close(df_multi_actifs: pd.DataFrame):
    figure=get_plotly_figure("Evolution du cours de cloture ajusté","Temps (mois et année)","Cours de cloture ajusté (USD)")
    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        if key != "TOTAL":
            figure=add_trace_plotly(figure,df_actif.droplevel(0).T['adj_close'],key,"lines")

    return figure

def get_plot_rendement(df_multi_actifs: pd.DataFrame):
    figure=get_plotly_figure("Evolution du rendement par actif","Temps (mois/ trimestre / année)","Rendement (USD)")
    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        figure=add_trace_plotly(figure,df_actif.droplevel(0).T['rendement'],key,"lines")
    return figure

def get_plot_investissement(df_multi_actifs: pd.DataFrame):
    figure=get_plotly_figure("Evolution de l'investissement cumulé par actif","Temps (mois et année)","Investissement cumulé (USD)")
    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        figure=add_trace_plotly(figure,df_actif.droplevel(0).T['investissement_cumule'],key,"lines")
    return figure

def get_table_stats(df_stats: pd.DataFrame):
    # Format the data
    df_stats = df_stats.apply(
        lambda x: x.apply(lambda x: f"{x*100:.2f} %") if x.name != "ratio_sharpe" else x.apply(lambda x: f"{x:.3f}"),
        axis=1
    )

    headers = ['Mesure'] + list(df_stats.columns)
    cell_values = [df_stats.index.tolist()] + [df_stats[col].tolist() for col in df_stats.columns]

    # Create the table figure
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="rgba(207,255,36,1)", 
                    font=dict(color="white", size=12), 
                    align="center",
                    height=40
                ),
                cells=dict(
                    values=cell_values,
                    fill_color="black", 
                    font=dict(color="white"),  
                    align="center",
                    line_color="white",  
                    height=30
                )
            )
        ]
    )

    # Update the layout
    fig.update_layout(
        title="Mesures financières associées au portefeuille",
        title_font=dict(color="white"), 
        paper_bgcolor="black",  
    )

    return fig


def get_plot_prediction_rendement(df_multi_actifs: pd.DataFrame):
    figure = get_plotly_figure("Prédiction du rendement par régression linéaire, +- 1 et 2 écart-types", "Temps (mois et année)", "Investissement cumulé (USD)")

    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        rendement_series = df_actif.droplevel(0).T['rendement']
        figure = add_trace_plotly(figure, rendement_series, key, "lines", legendgroup=key, visible=(True if key == "TOTAL" else "legendonly"))

        linear_model = LinearRegression()
        # Convertir les index en Timestamps si nécessaire
        X = pd.Series(pd.to_datetime(rendement_series.index)).apply(lambda date: date_to_int(date)).to_numpy().reshape(-1, 1)
        y = rendement_series
        linear_model.fit(X, y)
        y_pred = linear_model.predict(X)
        ecart_type = np.std(rendement_series - y_pred)

        time_axis = int_to_date(X.reshape(-1))

        figure.add_trace(
            go.Scatter(
                x=time_axis,
                y=y_pred,
                mode='lines',
                line=dict(dash='solid'),
                name='Prédiction du modèle linéaire',
                legendgroup=key,
                showlegend=False,
                visible=(True if key == "TOTAL" else "legendonly")
            )
        )

        for nb_ecart_type in [1, 2]:
            constante_ecart_type = nb_ecart_type * ecart_type
            for ecart_type, neg_symbole in zip([-constante_ecart_type, constante_ecart_type], ["-", "+"]):
                figure.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=y_pred + ecart_type,
                        mode='lines',
                        line=dict(dash='dash'),
                        name=f"{neg_symbole}{nb_ecart_type} écart-type",
                        legendgroup=key,
                        showlegend=False,
                        visible=(True if key == "TOTAL" else "legendonly")
                    )
                )

    return figure