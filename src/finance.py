import datetime
import yfinance as yf
import pandas as pd
import matplotlib as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

BASE_DATE = pd.Timestamp(2000,1,1)
DATE_FORMAT = '%Y-%m-%d'

def date_to_int(date):
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    return ((date - pd.Timestamp(BASE_DATE)).days)

def int_to_date(int_value):
    return BASE_DATE + pd.to_timedelta(int_value, unit='D')

def get_data(ind, date_debut, date_fin):
    data = yf.download(ind, start=date_debut.strftime(DATE_FORMAT), end=date_fin.strftime(DATE_FORMAT), progress=False)
    return data


# Calculer la valeur de l'investissement cumulé pour chaque mois de la période comprise entre date_debut et date_fin
def get_investissement_cumule(montant_initial, montant_recurrent, nb_mois):
    """"
    Renvoie le tableau des investissement cumulés pour chaque mois de la période
    """
    return montant_initial+np.linspace(start=0, stop=nb_mois-1, num=nb_mois)*montant_recurrent


def get_evolution_nb_actions(serie_adj_close: pd.Series, montant_initial, montant_recurrent):
    """
    Calcule le nombre d'actions possédées chaque mois en fonction du montant initial et recurrent investi
    On considère que chaque mois, la totalité du montant disponible est investie
    Retourne une série du nombre d'actions possédées chaque mois
    """
    # Calcul du nombre d'actions achetées avec l'investissement initial
    nb_actions = montant_initial/serie_adj_close.iloc[0]
    liste_nb_actions=[montant_initial/serie_adj_close.iloc[0]]
    # Pour chaque investissement recurrent
    for idx_mois, adj_close in enumerate(serie_adj_close[1:]):
        nb_actions += montant_recurrent/adj_close
        liste_nb_actions.append(nb_actions)
    return pd.Series(liste_nb_actions,index=serie_adj_close.index)


def filter_first_day_of_the_month(serie_adj_close: pd.Series):
    if serie_adj_close.empty:
        raise ValueError("The series is empty. No data to process.")
    adj_close = serie_adj_close.values
    adj_close_list = list(adj_close.flatten())
    year_month = np.array([f"{index.year}-{index.month:02}" for index in serie_adj_close.index].copy())
    day = np.array([index.day for index in serie_adj_close.index].copy())
    adj_close_dataframe = pd.DataFrame({
        'Adj Close': adj_close_list,
        'annee_mois': year_month.copy(),
        'jour': day.copy(),
    })
    premier_jour_adj_close = adj_close_dataframe.groupby('annee_mois')['jour'].min()
    premier_jour_adj_close = premier_jour_adj_close.rename("premier_jour")
    adj_close_dataframe = adj_close_dataframe.join(premier_jour_adj_close, on="annee_mois")
    
    # Handle months with no valid first-day data
    adj_close_dataframe = adj_close_dataframe[
        adj_close_dataframe['jour'] == adj_close_dataframe['premier_jour']
    ]

    if adj_close_dataframe.empty:
        raise ValueError("Filtered DataFrame is empty. No valid first-day data.")
    
    return adj_close_dataframe["Adj Close"]


def get_rendement_actif_unique(actif: str, date_debut: datetime, date_fin: datetime, montant_initial: int, montant_recurrent: int):
    serie_adj_close = get_data(actif, date_debut, date_fin)["Adj Close"]
    filtered_series_adj_close = filter_first_day_of_the_month(serie_adj_close)
    evolution_nb_actions = get_evolution_nb_actions(
    filtered_series_adj_close, montant_initial, montant_recurrent)
    investissement_cumule=get_investissement_cumule(
            montant_initial, montant_recurrent, len(filtered_series_adj_close))
    rendements_dataframe=pd.DataFrame(data={
        "rendement": evolution_nb_actions*filtered_series_adj_close-investissement_cumule,
        "nb_actions": evolution_nb_actions,
        "adj_close": filtered_series_adj_close,
        "investissement_cumule": investissement_cumule,
    })

     # Calcul de la volatilité
    rendements_dataframe["volatilite"] = calcul_volatilite(filtered_series_adj_close)

    # Calcul du rendement moyen annuel
    rendements_annuels = (rendements_dataframe["adj_close"] / rendements_dataframe["adj_close"].iloc[0]) ** (1 / len(rendements_dataframe.index)) - 1
    rendements_dataframe["sharpe_ratio"] = rendements_annuels / rendements_dataframe["volatilite"]


    return rendements_dataframe

def get_rendement_multi_actif(liste_actifs: list, liste_pourcentage_actifs: list, date_debut: datetime, date_fin: datetime, montant_initial: int, montant_recurrent: int, frais_gestion: float, frequence_contributions: str):
    if np.sum(liste_pourcentage_actifs) != 100:
        raise ValueError("La somme des pourcentages d'investissement pour chaque actif doit être égale à 100")
    
    # Calculate contribution intervals
    frequency_map = {
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1
    }
    if frequence_contributions not in frequency_map:
        raise ValueError("Fréquence de contribution invalide. Choisissez parmi: 'monthly', 'quarterly', 'yearly'")
    contributions_per_year = frequency_map[frequence_contributions]

    dict_df_rendements = {}
    for actif, pourcentage_actif in zip(liste_actifs, liste_pourcentage_actifs):
        dict_df_rendements[actif] = get_rendement_actif_unique(
            actif=actif,
            date_debut=date_debut,
            date_fin=date_fin,
            montant_initial=montant_initial * pourcentage_actif / 100,
            montant_recurrent=(montant_recurrent / contributions_per_year) * pourcentage_actif / 100
        )
    
    df_rendements = pd.concat(dict_df_rendements, axis=1)
    df_rendements = df_rendements.dropna(axis=0)

    # Apply frais_gestion to rendements
    rendement_columns = [col for col in df_rendements.columns if "rendement" in col]
    for col in rendement_columns:
        df_rendements[col] = df_rendements[col] * (1 - frais_gestion / 100)  # Deduct management fees
    
    investissement_cumule_columns = [col for col in df_rendements.columns if "investissement_cumule" in col]
    total_col = pd.DataFrame(
        {
            ("TOTAL", "rendement"): df_rendements[rendement_columns].sum(axis=1),
            ("TOTAL", "investissement_cumule"): df_rendements[investissement_cumule_columns].sum(axis=1)
        },
        index=df_rendements.index
    )
    print("total_col:", total_col)
    df_rendements = pd.concat([df_rendements, total_col], axis=1)

    return df_rendements

def add_acwi_reference(df_multi_actifs: pd.DataFrame,date_debut: datetime, date_fin: datetime, montant_initial: int, montant_recurrent: int, frais_gestion: float, frequence_contributions: str):
    df_acwi=get_rendement_multi_actif(liste_actifs=['ACWI'], liste_pourcentage_actifs=[100], date_debut=date_debut, date_fin=date_fin, montant_initial=montant_initial, montant_recurrent=montant_recurrent,frais_gestion=frais_gestion, frequence_contributions=frequence_contributions)
    df_acwi=df_acwi.drop(labels="TOTAL",axis=1)
    return pd.concat((df_multi_actifs,df_acwi),axis=1)

def calcul_volatilite(adj_close: pd.Series):
    log_return = np.log(adj_close / adj_close.shift(1))
    log_return=log_return.dropna()
    volatilite=log_return.std()
    return volatilite

def calcul_cagr(rendements: pd.Series, investissement_cumule: pd.Series):
    nb_annees=len(rendements)/12
    investissement_initial=investissement_cumule.iloc[0]
    investissement_final=investissement_cumule.iloc[-1]+rendements.iloc[-1]
    cagr=(investissement_final/investissement_initial)**(1/nb_annees)-1
    return cagr

def calcul_ratio_sharpe(rendement_annuel,volatilite_anuelle,rendement_actif_sans_risque=0.02):
    ratio_sharpe=(rendement_annuel-rendement_actif_sans_risque)/volatilite_anuelle
    return ratio_sharpe

def get_stats_df(df_multi_actifs: pd.DataFrame):
    stats_data={}

    for column in [col for col in df_multi_actifs.columns.get_level_values(0).unique() if col !="TOTAL"]:
        df_actif=df_multi_actifs[column]

        volatilite_mensuelle=calcul_volatilite(df_actif['adj_close'])
        volatilite_annuelle=calcul_volatilite(df_actif['adj_close'])*np.sqrt(12)
        cagr=calcul_cagr(df_actif['rendement'],df_actif['investissement_cumule'])
        ratio_sharpe=calcul_ratio_sharpe(cagr,volatilite_annuelle)

        stats_data[column]={
            "volatilite_mensuelle": volatilite_mensuelle,
            "volatilite_annuelle": volatilite_annuelle,
            "cagr": cagr,
            "ratio_sharpe": ratio_sharpe
        }

    return pd.DataFrame(data=stats_data)
