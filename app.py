import os
from flask import Flask, jsonify, abort
from flask_cors import CORS
import nfl_data_py as nfl
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from urllib.error import HTTPError

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- CORS Configuration ---
# Allow requests from your frontend's domain. '*' is a fallback for development.
frontend_url = os.environ.get('FRONTEND_URL', '*') 
CORS(app, resources={r"/api/*": {"origins": frontend_url}})

# --- CONFIGURATION ---
THE_ODDS_API_KEY = os.environ.get('THE_ODDS_API_KEY')

# --- CACHING ---
cached_data = None
last_fetch_time = None
CACHE_DURATION_MINUTES = 30

# --- DATA FETCHING AND PROCESSING ---

def get_team_stats():
    """ Fetches team statistics using nfl_data_py, handling the offseason and start-of-season edge cases. """
    try:
        now = datetime.now()
        current_year = now.year
        df = pd.DataFrame() # Initialize an empty DataFrame

        # --- THIS IS THE FINAL, MOST ROBUST LOGIC ---
        # First, try to get the current year's data.
        try:
            app.logger.info(f"Attempting to fetch NFL stats for the {current_year} season.")
            # Load the FULL dataset without specifying columns to avoid KeyErrors on download.
            df = nfl.import_weekly_data([current_year])
        except HTTPError as e:
            if e.code == 404:
                app.logger.warning(f"Data for {current_year} not found (404 Error). This is normal before the season starts.")
                df = pd.DataFrame()
            else:
                raise
        
        # If the current season's data is empty, fall back to the previous season.
        if df.empty:
            last_year = current_year - 1
            app.logger.warning(f"No weekly data found for {current_year}. Falling back to {last_year} season data.")
            df = nfl.import_weekly_data([last_year])
            if df.empty:
                app.logger.error(f"CRITICAL: No weekly data found for {last_year} either. Cannot provide stats.")
                return None

        app.logger.info(f"Successfully loaded data for the {df['season'].iloc[0]} season.")
        
        # Now that data is loaded, safely select the columns we need.
        required_cols = ['team', 'season', 'week', 'result', 'spread_line', 'points_for', 'points_against']
        df = df[required_cols]

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
        app.logger.error(f"CRITICAL ERROR in get_team_stats: {e}", exc_info=True)
        return None

def get_nfl_odds():
    """ Fetches live NFL odds from The Odds API. """
    api_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={THE_ODDS_API_KEY}&regions=us&markets=spreads,h2h&oddsFormat=american"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        app.logger.error(f"CRITICAL ERROR in get_nfl_odds: {e}", exc_info=True)
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

# --- API ENDPOINT ---
@app.route('/api/nfl-predictions')
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
        app.logger.error(f"Failed to fetch data. Stats fetched: {'Yes' if team_stats else 'No'}. Odds fetched: {'Yes' if odds_data else 'No'}")
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

# --- SERVE A SIMPLE HEALTH CHECK at root ---
@app.route('/')
def health_check():
    """ A simple health check to confirm the server is running. """
    return "Backend is running."

