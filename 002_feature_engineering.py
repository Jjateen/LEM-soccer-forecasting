import pandas as pd
import numpy as np
import json

from glob import glob

FIELD_SIZE = {'x':1.05, 'y':0.68}

df = pd.read_csv('data\wyscout\csv\events\England.csv')
df

subevent_type_map = {
    'air_duel': 1,
    'ground_attacking_duel': 2,
    'ground_defending_duel': 3,
    'ground_loose_ball_duel': 4,
    'foul': 5,
    'hand_foul': 6,
    'late_card_foul': 7,
    'out_of_game_foul': 8,
    'protest': 9,
    'simulation': 10,
    'time_lost_foul': 11,
    'violent_foul': 12,
    'corner': 13,
    'free_kick': 14,
    'free_kick_cross': 15,
    'goal_kick': 16,
    'penalty': 17,
    'throw_in': 18,
    'goalkeeper_leaving_line': 19,
    'acceleration': 20,
    'clearance': 21,
    'touch': 22,
    'cross': 23,
    'hand_pass': 24,
    'head_pass': 25,
    'high_pass': 26,
    'launch': 27,
    'simple_pass': 28,
    'smart_pass': 29,
    'reflexes': 30,
    'save_attempt': 31,
    'free_kick_shot': 32,
    'shot': 33,
}

event_type_map = {
    'duel': 1,
    'foul': 2,
    'free_kick': 3,
    'goalkeeper_leaving_line': 4,
    'offside': 5,
    'others_on_the_ball': 6,
    'pass': 7,
    'interruption': 8,
    'save_attempt': 9,
    'shot': 10,
}

df['subtype_id'] = df['subtype_name'].map(subevent_type_map)
df['type_id'] = df['type_name'].map(event_type_map)

df.fillna(0)[['type_name', 'subtype_name', 'subtype_id']].value_counts()

# A possession starts with a pass and ends when a successful pass from the opponent is made
# or when the ball goes out of play
start_new_possession = (((df['type_name'] == 'pass') * df['accurate'] + (df['type_name'] == 'free_kick')) * df.team_id).replace(0, np.NaN).fillna(method='ffill')
start_new_possession = (start_new_possession != start_new_possession.shift(1)).cumsum()
start_new_possession = start_new_possession + ((df['type_name'] == 'interruption') | (df['type_name'] == 'foul')).shift(1).fillna(0).cumsum()
df['possession_id'] = start_new_possession
df['possession_type_name'] = (df['possession_id'].diff(1).fillna(1) * df['type_name']).replace('', np.NaN).fillna(method='ffill')
df['possession_type_id'] = df['possession_type_name'].map(event_type_map)
df['possession_team_id'] = (df['possession_id'].diff(1).fillna(1) * df['team_id']).replace(0, np.NaN).fillna(method='ffill')
df['possession_start_time'] = (df['possession_id'].diff(1).fillna(1) * df['absolute_sec']).replace(0, np.NaN).fillna(method='ffill')

for i in range(1, 3):
    df[f'previous_action_type_id_{i}'] = df['type_id'].shift(i)
    df[f'previous_action_is_same_team_{i}'] = (df['team_id'] == df['team_id'].shift(i)).astype(int)
    df[f'previous_action_is_same_possession_{i}'] = (df['possession_id'] == df['possession_id'].shift(i)).astype(int)
    df[f'previous_action_is_same_player_{i}'] = (df['player_id'] == df['player_id'].shift(i)).astype(int)
    df[f'previous_action_x_{i}'] = abs((100 * (1-df[f'previous_action_is_same_team_{i}'])) - df['x'].shift(i))
    df[f'previous_action_y_{i}'] = abs((100 * (1-df[f'previous_action_is_same_team_{i}'])) - df['y'].shift(i))
    df[f'previous_action_time_since_{i}'] = df['absolute_sec'] - df['absolute_sec'].shift(i)
    df[f'previous_action_x_displacement_{i}'] = df['x'] - df[f'previous_action_x_{i}']

df['possession_start_is_same_team'] = (df['possession_team_id'] == df['team_id']).astype(int)
df['possession_start_action_x'] = (df['possession_id'].diff(1).fillna(1) * df['x']).replace(0, np.NaN).fillna(method='ffill')
df['possession_start_action_y'] = (df['possession_id'].diff(1).fillna(1) * df['y']).replace(0, np.NaN).fillna(method='ffill')
df['possession_start_time_since'] = df['absolute_sec'] - df['possession_start_time']
df['possession_start_x_displacement'] = df['x'] - df['possession_start_action_x']

df['start_distance_to_goal'] = np.sqrt(((df['x'] - 100) * FIELD_SIZE['x'])**2 + ((df['y'] - 50) * FIELD_SIZE['y'])**2)
df['start_angle_to_goal'] = abs(np.arctan2((df['y'] - 50) * FIELD_SIZE['y'], (df['x'] - 100) * FIELD_SIZE['x']))
df['end_distance_to_goal'] = np.sqrt(((df['end_x'] - 100) * FIELD_SIZE['x'])**2 + ((df['end_y'] - 50) * FIELD_SIZE['y'])**2)
df['end_angle_to_goal'] = abs(np.arctan2((df['end_y'] - 50) * FIELD_SIZE['y'], (df['end_x'] - 100) * FIELD_SIZE['x']))

df['intent_progressive'] = ((df['type_name'] == 'pass') * (df['end_distance_to_goal'] < df['start_distance_to_goal'])).astype(int)

df['shot_assist'] = (((df['type_name'].isin(['pass', 'free_kick']) & (df['accurate'] == 1)) & ((df['type_name'].shift(1) == 'shot') | (df['type_name'].shift(2) == 'shot'))).diff() < 0).shift(-1).fillna(0).astype(int)

df['goal'] = df['goal'].fillna(0)

actions_before_goal = None
actions_before_own_goal = None
for i in range(10):
    if actions_before_goal is None:
        actions_before_goal = df.goal.shift(-(i))
        actions_before_own_goal = -df.own_goal.shift(-(i))
    else:
        actions_before_goal += df.goal.shift(-(i))
        actions_before_own_goal -= df.own_goal.shift(-(i))
actions_before_goal = actions_before_goal.fillna(0)
actions_before_own_goal = actions_before_own_goal.fillna(0)

is_same_period = (df.goal * df.period).replace(to_replace=False, method='bfill') == df.period
is_same_game = (df.goal * df.match_id).replace(to_replace=False, method='bfill') == df.match_id
is_team_next_goal = 2 * ((df.goal * df.team_id).replace(to_replace=False, method='bfill') == df.team_id) - 1
is_team_next_goal *= actions_before_own_goal

df['vaep_label_0'] = actions_before_goal * is_same_period * is_same_game * is_team_next_goal
df['vaep_label_0_scoring'] = df['vaep_label_0'].clip(0, 1)
df['vaep_label_0_conceding'] = abs(df['vaep_label_0'].clip(-1, 0))

print("Added New Features")