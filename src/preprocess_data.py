# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

#실패한 패스 식별방법
def identify_pass(row):
    """
    Identify passes from the data
    """
    if type(row["subEventName"]) == float:
        return row["eventName"]
    elif (
        row["eventName"] == "Ball lost"
        and "interception" in row["subEventName"].lower()
    ):
        return "Pass"
    else:
        return row["eventName"]

#패스에성공 여부에 대한 레이블링 부여함
def compute_wyscout_columns(df_events):
    """
    Function computes all columns that are needed in order to fit with the Wyscout format
    :param df_events: (pd.DataFrame) Data frame containing all event data
    :param game: (int) Id of the game
    :return: pd.DataFrame containing all relevant columns needed for the Wyscout format
    """

    # set the teamId - home team is always 1, away team always 2
    df_events["teamId"] = np.where(df_events["team"] == "Home", 1, 2)

    # make sure that eventName and subEventName are written lower case
    df_events["eventName"] = df_events["type"].map(lambda x: x[0] + x[1:].lower())
    df_events["subEventName"] = df_events["subtype"].map(lambda x: x[0] + x[1:].lower())

    # add a columns indicating that a pass was accurate
    # type이 Pass인 데이터는 모두 패스가 성공한 데이터
    # type = "BALL LOST" : 패스가 실패한 데이터
    # subtype in "INTERCEPTION" : 패스가 실패한 데이터(헤딩, 크롯, 골킥등 다양한 인터셉트는 실패한 패스임)
    df_events["accurate"] = 1 * (df_events["eventName"] == "Pass")

    # compute the event name
    df_events["eventName"] = df_events.apply(identify_pass, axis=1)
    df_events["eventName"] = np.where(
        df_events["eventName"] == "Challenge", "Duel", df_events["eventName"]
    )

    return df_events

#패스에 대한 레이블링을 부여함(성공여부, 골여부, 자책골 여부)
def cleanse_metrica_event_data(df_events):
    """
    Function to clean the Metrica event data. Notice that quite a lot of the code is needed to make the Metrica data
    compatible with the Wyscout format
    :param game: (int) GameId
    :param reverse: (bool) If True, the away team is playing left to right in the first half
    :return: None
    """

    # identify goals and own goals
    df_events["goal"] = 1 * (
        df_events.apply(
            lambda row: row["type"] == "SHOT" and "-GOAL" in row["subtype"], axis=1
        )
    )
    df_events["ownGoal"] = 1 * (
        df_events.apply(
            lambda row: row["type"] == "BALL OUT" and "-GOAL" in row["subtype"], axis=1
        )
    )

    #패스 성공 여부에 대한 레이블링 부여함
    df_events = compute_wyscout_columns(df_events)

    return df_events

#soccermap형식으로 변환하기 위해서 하나의 컬럼(freeze_frame)에 선수(key)별 정보(value)방식으로 저장해놓음
#변환할 때, 코드를 단축해주고 시간을 단축해주기 때문에 사용
def make_freeze_frame(events, tracks):
    events['freeze_frame'] = dict()

    for i in tqdm(range(len(events))):
        start_frame, end_frame = events.loc[i,['start_frame','end_frame']]
        frame_track = tracks[(tracks['frame'] >= start_frame) & (tracks['frame']  <= end_frame)]

        #tracking-data를 수집할 수 없는 이벤트는 제외
        if frame_track.empty:
            events.at[i,'freeze_frame'] = {}
            continue

        #frame에 해당하는 인덱스는 어차피 한개
        start_index = frame_track.index[0]
        end_index = frame_track.index[-1]

        freeze_dict = {}
        
        #이벤트를 수행하는 선수과 같은팀 선수 / 상대팀 선수를 분류
        event_player = events.at[i,'from']
        team = event_player[0]

        teammates = [c[:-2] for c in frame_track.dropna(axis=1).columns if c[0] == team and c.endswith("_x")]
        opoonents = [c[:-2] for c in frame_track.dropna(axis=1).columns if c[0] != team and c[:4] != "ball" and c.endswith("_x")]

        for col in teammates:
            teammate = True
            actor = True if col == event_player else False
            ball = False

            start_x = frame_track.at[start_index, f'{col}_x']
            start_y = frame_track.at[start_index, f'{col}_y']
            start_vx = frame_track.at[start_index, f'{col}_vx']
            start_vy = frame_track.at[start_index, f'{col}_vy']
            start_speed = frame_track.at[start_index, f'{col}_speed']

            end_x = frame_track.at[end_index, f'{col}_x']
            end_y = frame_track.at[end_index, f'{col}_y']
            end_vx = frame_track.at[end_index, f'{col}_vx']
            end_vy = frame_track.at[end_index, f'{col}_vy']
            end_speed = frame_track.at[end_index, f'{col}_speed']

            teammate_info_dict = {'teammate': teammate,'actor':actor,'ball':ball,'start_x':start_x,'start_y':start_y,'start_vx':start_vx,'start_vy':start_vy,
                                                                        'end_x':end_x,'end_y':end_y,'end_vx':end_vx,'end_vy':end_vy}

            freeze_dict[col] = teammate_info_dict   

        for col in opoonents:
            teammate = False
            actor = False
            ball = False

            start_x = frame_track.at[start_index, f'{col}_x']
            start_y = frame_track.at[start_index, f'{col}_y']
            start_vx = frame_track.at[start_index, f'{col}_vx']
            start_vy = frame_track.at[start_index, f'{col}_vy']
            start_speed = frame_track.at[start_index, f'{col}_speed']

            end_x = frame_track.at[end_index, f'{col}_x']
            end_y = frame_track.at[end_index, f'{col}_y']
            end_vx = frame_track.at[end_index, f'{col}_vx']
            end_vy = frame_track.at[end_index, f'{col}_vy']
            end_speed = frame_track.at[end_index, f'{col}_speed']

            opoonent_info_dict = {'teammate': teammate,'actor':actor,'ball':ball,'start_x':start_x,'start_y':start_y,'start_vx':start_vx,'start_vy':start_vy,
                                                                        'end_x':end_x,'end_y':end_y,'end_vx':end_vx,'end_vy':end_vy}
            freeze_dict[col] = opoonent_info_dict  

        for col in ["ball"]:
            teammate = False
            actor = False
            ball = True

            start_x = frame_track.at[start_index, f'{col}_x']
            start_y = frame_track.at[start_index, f'{col}_y']

            end_x = frame_track.at[end_index, f'{col}_x']
            end_y = frame_track.at[end_index, f'{col}_y']

            ball_info_dict = {'teammate': teammate,'actor':actor,'ball':ball,'start_x':start_x,'start_y':start_y,'start_vx':None,'start_vy':None,
                                                                        'end_x':end_x,'end_y':end_y,'end_vx':None,'end_vy':None}
            

            freeze_dict[col] = ball_info_dict   

        events.at[i,'freeze_frame'] = freeze_dict

    return events

#항상 home팀이 왼쪽 & away팀이 오른쪽에 배치시키도록함
def rotate_pitch(reverse, events, field_dimen):
    session = 1 if reverse else 2
    events.loc[events['session'] == session,'freeze_frame'] = events.loc[events['session'] == session].apply(lambda row: freeze_left_to_right(row, field_dimen), axis=1)

    return events

def freeze_left_to_right(actions, field_dimen):
    freezedf = pd.DataFrame.from_records(actions["freeze_frame"])

    for player in freezedf.keys():
        freezedf[player]["start_x"] = field_dimen[0] - freezedf[player]["start_x"]
        freezedf[player]["start_y"] = field_dimen[1] - freezedf[player]["start_y"]     
        freezedf[player]["end_x"] = field_dimen[0] - freezedf[player]["end_x"]
        freezedf[player]["end_y"] = field_dimen[1] - freezedf[player]["end_y"]     

        if player != 'ball':
            freezedf[player]["start_vx"] = -freezedf[player]["start_vx"]
            freezedf[player]["start_vy"] = -freezedf[player]["start_vy"]       
            freezedf[player]["end_vx"] = -freezedf[player]["end_vx"]
            freezedf[player]["end_vy"] = -freezedf[player]["end_vy"]      

    return freezedf.to_dict()   

def preprocess_data(game_id, save_folder=None,field_dimen=(106,68)):
    # 전반이든 후반이든 하상 home -> away로 단일방향으로 설정함(추후에 공격방향이 왼쪽에 배치시킬 때 유용하게 활용함)
    # metrica1 : 전반전 : home(A) - away(B)
    # 	         후반적 : away(B) - home(A)

    # metrica2 : 전반전 :  away(B) - hOME(A)
    # 	         후반적 :  home(A) - away(B)

    # metrica3 : 전반전 :  home(B) - away(A)
    # 	         후반적 :  away(A) - home(B)

    #metrica1, 3경기는 후반전을 flip하고, 2경기는 전반전을 flip함
    reverse_dict = {'1':False, '2':True, '3':False, '3_valid':False, '3_test':False}

    events = pd.read_csv(f'../metrica-data/event-data/match{game_id}.csv')
    tracks = pd.read_csv(f'../metrica-data/tracking-data/match{game_id}.csv')

    events = cleanse_metrica_event_data(events)
    events = make_freeze_frame(events, tracks)
    events = rotate_pitch(reverse_dict[game_id], events, field_dimen)

    if save_folder:
        events.to_csv(f'{save_folder}/match{game_id}.csv',index=False)

    return events, tracks