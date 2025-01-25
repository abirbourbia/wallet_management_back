from flask import Flask, jsonify, request
from flask_cors import CORS
from src.finance import get_rendement_multi_actif, get_stats_df,add_acwi_reference, get_stats_df
from src.date_utils import string_to_date
from src.plot import get_plot_adj_close, get_plot_histogram, get_plot_rendement,get_plot_investissement, get_table_stats,get_plot_prediction_rendement,convert_plotly_to_json

app = Flask(__name__)
CORS(app)

def process_data(data):
    """Helper function to process input data."""
    date_debut = string_to_date(data['dateDebut'])
    date_fin = string_to_date(data['dateFin'])
    montant_initial = int(data['investInit'])
    montant_recurrent = int(data['investRecu'])
    actifs = [actif['etf'] for actif in data['listActifs']]
    pourcentages_actifs = [int(actif['percentage']) for actif in data['listActifs']]
    frais_gestion = float(data['fraisGestion'])
    frequence_contributions = data['frequenceContributions']

    df_multi_actifs = get_rendement_multi_actif(
        liste_actifs=actifs,
        liste_pourcentage_actifs=pourcentages_actifs,
        date_debut=date_debut,
        date_fin=date_fin,
        montant_initial=montant_initial,
        montant_recurrent=montant_recurrent,
        frais_gestion=frais_gestion,
        frequence_contributions=frequence_contributions
    )
    return df_multi_actifs

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify({'message': 'Hello from the back!'})

@app.route('/get_all_data', methods=['POST'])
def get_all_data():
    data = request.get_json()
    df_multi_actifs = process_data(data)
    
    # Generate statistics DataFrame
    df_stats = get_stats_df(df_multi_actifs)
    
    # Add ACWI reference and process
    df_multi_actifs = add_acwi_reference(
        df_multi_actifs=df_multi_actifs,
        date_debut=string_to_date(data['dateDebut']),
        date_fin=string_to_date(data['dateFin']),
        montant_initial=int(data['investInit']),
        montant_recurrent=int(data['investRecu']),
        frais_gestion=float(data['fraisGestion']),
        frequence_contributions=data['frequenceContributions']
    ).dropna()

    # Generate figures
    figures = {
        "1.1-plot_rendement": get_plot_rendement(df_multi_actifs),
        "1.2-plot_histogram": get_plot_histogram(df_multi_actifs),
        "2-plot_investissement": get_plot_investissement(df_multi_actifs.drop(labels="ACWI", axis=1)),
        "3-table_stats": get_table_stats(df_stats),
        "4-plot_prediction_rendement": get_plot_prediction_rendement(df_multi_actifs),
        "5-adj_close_plot": get_plot_adj_close(df_multi_actifs),
    }

    # Convert figures to JSON
    json_figures = {key: convert_plotly_to_json(plot) for key, plot in figures.items()}
    
    # Convert df_stats to JSON-compatible format
    df_stats_json = df_stats.to_dict(orient='records')  # Convert to a list of dictionaries
    
    # Return both figures and stats
    return jsonify({'success': 'true', 'json_figures': json_figures, 'df_stats': df_stats_json})


@app.route('/metrics', methods=['POST'])
def get_metrics():
    data = request.get_json()
    df_multi_actifs = process_data(data)
    df_stats = get_stats_df(df_multi_actifs)
    stats_json = df_stats.to_dict(orient='records')  # Convert DataFrame to JSON
    return jsonify({'success': 'true', 'stats': stats_json})

if __name__ == '__main__':
    app.run(debug=True)