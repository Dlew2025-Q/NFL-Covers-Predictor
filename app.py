import os
from flask import Flask, jsonify, send_from_directory, abort, Blueprint
from flask_cors import CORS
import nfl_data_py as nfl
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)

# --- Initialize Flask App ---
app = Flask(__name__, static_folder='static')
CORS(app)

# --- CONFIGURATION ---
THE_ODDS_API_KEY = os.environ.get('THE_ODDS_API_KEY')

# --- CACHING ---
cached_data = None
last_fetch_time = None
CACHE_DURATION_MINUTES = 30

# --- API BLUEPRINT (for better route organization) ---
api_bp = Blueprint('api', __name__, url_prefix='/api')

# --- DATA FETCHING AND PROCESSING ---
def get_team_stats():
    """ Fetches the latest team statistics using nfl_data_py. """
    try:
        current_year = datetime.now().year
        cols = ['team', 'season', 'week', 'result', 'spread_line', 'points_for', 'points_against']
        df = nfl.import_weekly_data([current_year], columns=cols)
        
        df['ats_result'] = 'push'
        df.loc[df['result'] + df['spread_line'] > 0, 'ats_result'] = 'win'
        df.loc[df['result'] + df['spread_line'] < 0, 'ats_result'] = 'loss'

        team_stats = df.groupby('team').agg(
            ppg=('points_for', 'mean'),
            opp_ppg=('points_against', 'mean'),
            ats_wins=('ats_result', lambda x: (x == 'win').sum()),
            ats_losses=('ats_result', lambda x: (x == 'loss').sum()),
            ats_pushes=('ats_result', lambda x: (x == 'push').sum())
        ).reset_index()
        
        team_stats_dict = team_stats.set_index('team').to_dict('index')
        
        team_mapping = nfl.import_team_desc()[['team_abbr', 'team_name']]
        abbr_to_name = dict(zip(team_mapping['team_abbr'], team_mapping['team_name']))
        
        final_stats = {abbr_to_name[abbr]: data for abbr, data in team_stats_dict.items() if abbr in abbr_to_name}
        return final_stats
    except Exception as e:
        app.logger.error(f"Error fetching stats from nfl_data_py: {e}")
        return None

def get_nfl_odds():
    """ Fetches live NFL odds from The Odds API. """
    api_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={THE_ODDS_API_KEY}&regions=us&markets=spreads,h2h&oddsFormat=american"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching odds from The Odds API: {e}")
        return None

def transform_api_data(api_games):
    """ Transforms raw odds data into a clean list of game objects. """
    games = []
    if not api_games: return games
    for game in api_games:
        bookmaker = game.get('bookmakers', [])[0] if game.get('bookmakers') else None
        if not bookmaker: continue
        
        spreads_market = next((m for m in bookmaker.get('markets', []) if m['key'] == 'spreads'), None)
        if not spreads_market or len(spreads_market.get('outcomes', [])) < 2: continue

        favorite_outcome = next((o for o in spreads_market['outcomes'] if o['point'] < 0), None)
        if not favorite_outcome: continue
        
        games.append({
            'id': game['id'], 'gameTime': game['commence_time'], 'awayTeam': game['away_team'],
            'homeTeam': game['home_team'], 'favorite': favorite_outcome['name'], 'line': favorite_outcome['point']
        })
    return games

def calculate_cover_probability(game, all_team_stats):
    """ Calculates the estimated cover probability for the favorite. """
    home_stats = all_team_stats.get(game['homeTeam'])
    away_stats = all_team_stats.get(game['awayTeam'])
    if not home_stats or not away_stats: return 50.0
    home_power = home_stats['ppg'] - home_stats['opp_ppg']
    away_power = away_stats['ppg'] - away_stats['opp_ppg']
    projected_spread = away_power - home_power - 2.5
    actual_line = game['line'] if game['favorite'] == game['homeTeam'] else -game['line']
    value_difference = projected_spread - actual_line
    probability = 50 + (value_difference * 2.5)
    return max(5, min(95, probability))

# --- API ENDPOINT (now attached to the Blueprint) ---
@api_bp.route('/nfl-predictions')
def get_nfl_predictions():
    """ The main API endpoint that combines all data and returns predictions. """
    global cached_data, last_fetch_time

    if cached_data and last_fetch_time and (datetime.utcnow() - last_fetch_time) < timedelta(minutes=CACHE_DURATION_MINUTES):
        app.logger.info("Returning cached data.")
        return jsonify(cached_data)

    app.logger.info("Fetching new data from APIs.")
    if not THE_ODDS_API_KEY:
        app.logger.error("THE_ODDS_API_KEY environment variable not set.")
        abort(500, description="API key is not configured on the server.")

    team_stats = get_team_stats()
    odds_data = get_nfl_odds()

    if not team_stats or not odds_data:
        abort(503, description="Failed to fetch data from one or more external sources.")

    all_games = transform_api_data(odds_data)
    
    today = datetime.utcnow()
    days_since_thursday = (today.weekday() - 3) % 7
    start_of_week = (today - timedelta(days=days_since_thursday)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = start_of_week + timedelta(days=7)
    
    current_week_games = [
        game for game in all_games 
        if start_of_week <= datetime.fromisoformat(game['gameTime'].replace('Z', '+00:00')) < end_of_week
    ]

    predictions = []
    for game in current_week_games:
        probability = calculate_cover_probability(game, team_stats)
        favorite_stats = team_stats.get(game['favorite'])
        ats_record = "N/A"
        if favorite_stats:
            ats_record = f"{int(favorite_stats['ats_wins'])}-{int(favorite_stats['ats_losses'])}"
            if favorite_stats['ats_pushes'] > 0:
                 ats_record += f"-{int(favorite_stats['ats_pushes'])}"

        predictions.append({
            **game, 'cover_probability': round(probability, 1), 'favorite_ats_record': ats_record
        })
    
    predictions.sort(key=lambda x: x['gameTime'])

    cached_data = predictions
    last_fetch_time = datetime.utcnow()
    app.logger.info(f"Successfully fetched and processed {len(predictions)} predictions.")
    return jsonify(predictions)

# Register the Blueprint with the main Flask app
app.register_blueprint(api_bp)

# --- SERVE FRONTEND (This is the corrected, simpler route) ---
@app.route('/')
def serve_index():
    """ Serves the main index.html file for the root URL. """
    return send_from_directory(app.static_folder, 'index.html')

