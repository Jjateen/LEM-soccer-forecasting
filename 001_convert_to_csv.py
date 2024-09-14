import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import json
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_wyscout_json(path):
    with open(path, encoding='latin-1') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def fix_positions(df):
    df['x'] = df['positions'].apply(lambda x: x[0]['x']).clip(0, 100)
    df['y'] = df['positions'].apply(lambda x: 100 - x[0]['y']).clip(0, 100)
    df['end_x'] = df['positions'].apply(lambda x: x[1]['x'] if len(x) > 1 else np.NaN)
    df['end_y'] = df['positions'].apply(lambda x: 100 - x[1]['y'] if len(x) > 1 else np.NaN)
    df = df.drop(columns=['positions'])
    return df

def fix_tags(df):
    tags_map = {
        101: 'goal', 102: 'own_goal', 301: 'assist', 302: 'key_pass',
        1901: 'counter_attack', 401: 'left', 402: 'right', 403: 'head',
        1101: 'direct', 1102: 'indirect', 2001: 'dangerous_ball_lost',
        2101: 'blocked', 801: 'high', 802: 'low', 1401: 'interception',
        1501: 'clearance', 201: 'opportunity', 1301: 'feint',
        1302: 'missed ball', 501: 'free_space_r', 502: 'free_space_l',
        503: 'take_on_l', 504: 'take_on_r', 1601: 'sliding_tackle',
        601: 'anticipated', 602: 'anticipation', 1701: 'red',
        1702: 'yellow', 1703: 'second_yellow', 901: 'through',
        1001: 'fairplay', 701: 'lost', 702: 'neutral', 703: 'won',
        1801: 'accurate', 1802: 'not_accurate',
    }

    tags_goal_mouth_map = {
        1201: 'gb', 1202: 'gbr', 1203: 'gc', 1204: 'gl', 1205: 'glb',
        1206: 'gr', 1207: 'gt', 1208: 'gtl', 1209: 'gtr', 1210: 'obr',
        1211: 'ol', 1212: 'olb', 1213: 'or', 1214: 'ot', 1215: 'otl',
        1216: 'otr', 1217: 'pbr', 1218: 'pl', 1219: 'plb', 1220: 'pr',
        1221: 'pt', 1222: 'ptl', 1223: 'ptr',
    }

    df['tags'] = df['tags'].apply(lambda x: [t['id'] for t in x])
    for tag_id, tag_name in tags_map.items():
        df[tag_name] = df['tags'].apply(lambda x: 1 if tag_id in x else np.NaN)

    df['goal_mouth_placement'] = ''
    for tag_id, placement in tags_goal_mouth_map.items():
        df['goal_mouth_placement'] += df['tags'].apply(lambda x: placement * (tag_id in x))
    
    df = df.drop(columns=['tags'])
    return df

def fix_time_variables(df):
    df['absolute_sec'] = df['eventSec'].apply(lambda x: x).round(1)
    df['minute'] = df['eventSec'].apply(lambda x: int(x/60))
    df['second'] = df['eventSec'].apply(lambda x: int(x%60))
    df['period'] = df['matchPeriod'].map({'1H': 1, '2H': 2, 'E1': 3, 'E2': 4})
    df = df.drop(columns=['eventSec', 'matchPeriod'])
    return df 

def fix_type_names(df):
    df['type_name'] = df['eventName'].str.replace(' ', '_').str.lower()
    df['subtype_name'] = df['subEventName'].str.replace(' ', '_').str.lower()
    df = df.drop(columns=['eventName', 'eventId', 'subEventName', 'subEventId'])
    return df

def fix_club_names(df):
    df_teams = load_wyscout_json('data/wyscout/json/teams.json')
    df['wyId'] = df['teamId']
    df = df.merge(df_teams[['wyId', 'name']], on='wyId', how='left').rename(columns={'wyId': 'team_id', 'name': 'team_name'})
    df = df.drop(columns=['teamId'])
    return df

def fix_player_names(df):
    df_players = load_wyscout_json('data/wyscout/json/players.json')
    df['wyId'] = df['playerId']
    df = df.merge(df_players[['wyId', 'shortName']], on='wyId', how='left').rename(columns={'wyId': 'player_id', 'shortName': 'player_name'})
    df = df.drop(columns=['playerId'])
    return df

def fix_match_names(df, competition):
    df_matches = load_wyscout_json(os.path.join('data', 'wyscout', 'json', 'matches', f'{competition}.json'))
    home_team_id = []
    away_team_id = []
    for _, row in df_matches.iterrows():
        teams = list(row['teamsData'].keys())
        if row['teamsData'][teams[0]]['side'] == 'home':
            home_team_id.append(teams[0])
            away_team_id.append(teams[1])
        else:
            home_team_id.append(teams[1])
            away_team_id.append(teams[0])
    df_matches['label'] = df_matches['label'].str.split(',').str[0]
    df_matches['home_team_id'] = home_team_id
    df_matches['away_team_id'] = away_team_id
    df['wyId'] = df['matchId']
    df = df.merge(df_matches[['wyId', 'label', 'winner', 'home_team_id', 'away_team_id']], on='wyId', how='left').rename(columns={'wyId': 'match_id', 'label': 'match_name', 'winner': 'match_winner'})
    df = df.drop(columns=['matchId'])
    return df

def load_wyscout_events_json(competition):
    file_path = os.path.join('data', 'wyscout', 'json', 'events', f'{competition}.json')
    print(f"Attempting to open file: {file_path}")
    df = load_wyscout_json(file_path)
    df = fix_match_names(df, competition)
    df = fix_time_variables(df)
    df = fix_player_names(df)
    df = fix_club_names(df)
    df = fix_positions(df)
    df = fix_type_names(df)
    df = fix_tags(df)
    return df

def convert_wyscout_events(competition):
    df = load_wyscout_events_json(competition)
    output_path = os.path.join('data', 'wyscout', 'csv', 'events', f'{competition}.csv')
    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV file: {output_path}")

# Main execution
events_path = os.path.join('data', 'wyscout', 'json', 'events', '*.json')
total_files = len(glob(events_path))

for i, competition_path in enumerate(glob(events_path)):
    competition = os.path.basename(competition_path).split('.')[0]
    print(f'{i+1}/{total_files} - Converting {competition}.' + '.' * 50)
    convert_wyscout_events(competition)

print("\nConversion complete!")