#1. 모든 액션의 가치를 레이블링해야한다
#2. 소유권이란 session이 바뀌거나 득점이 된 경우를 의미한다
#3. 득점/실점을 할 경우 이전 소유권(session의 시작 / 이전 득실점)과 현재 골 사이의 모든 행동들은 Attacking_team=1, defending_team=-1를 받는다
#4. 시간 앱실론 = 15초 : 단기적 가치과 장기적 가치의 균형을 마치기 위해서 사용되는 시간상수로 현재 골 이전의 15초동안 발생한 행동만 보상을 받는다
#5. 우리는 세트피스동안 발생하는 행동(프리킥, 코너킥, 페널티킥등)은 분석에서 제외한다 -> type=SET PIECE인 행동

import pandas as pd
import numpy as np

def get_value_labels(events):
    events['value_label'] = 0

    for i in range(len(events)):
        current_session = events.at[i, 'session']
        current_time = events.at[i, 'end_time']
        current_team = events.at[i, 'team']
        epsilon = 15

        future_events = events[(events['end_time'] >current_time) & (events['end_time'] <= current_time+epsilon)]

        future_goal_indices = future_events[(future_events['goal']==1) | (future_events['ownGoal']==1)].index.to_list()

        if future_goal_indices:
            #만약 득/실점한 상황이 두개라면? -> 즉 득점한 후에 킥오프하자마자 실점을 했다면?
            #이럴경우 때문에 우리는 무조건 전자의 득점상황에 대한 레이블링을 수행해야함
            future_close_goal_indices = future_goal_indices[0]
            future_session= events.at[future_close_goal_indices, 'session']
            future_team = events.at[future_close_goal_indices, 'team']

            score_condition = (current_session == future_session) & (current_team == future_team)
            concede_condition = (current_session == future_session) & (current_team != future_team)

            if score_condition:
                events.at[i,'value_label'] = 1
            if concede_condition:
                events.at[i,'value_label'] = -1

    return events

def get_intended_receiver(events):
    events['Intended_Receiver'] = [{} for _ in range(len(events))]

    for idx, action in events.iterrows():
        Intended_dict = {}
        if not action["freeze_frame"]:
                continue
        
        frame = pd.DataFrame.from_dict(action["freeze_frame"],orient='index')

        teammate_frame = frame.loc[~frame.actor & frame.teammate, ["start_x", "start_y"]]

        receiver_coo = teammate_frame.values.reshape(-1, 2)

        #각 팀원 선수의 인덱스 -> 선수ID를 매핑하는 딕셔너리 : receiver ID찾는 작업
        player_index_dict = {index:player for index, player in enumerate(teammate_frame.index)}

        ball_coo = frame.loc[frame.ball,['start_x', 'start_y']].values.reshape(-1, 2)
        interception_coo = frame.loc[frame.ball,['end_x', 'end_y']].values.reshape(-1,2)

        dist = np.sqrt((receiver_coo[:, 0] - interception_coo[:, 0]) ** 2+ (receiver_coo[:, 1] - interception_coo[:, 1]) ** 2)
        exp_receiver_dist = np.argmax((np.amin(dist) / dist))

        Intended_dict['dist'] = {'ID':player_index_dict[exp_receiver_dist], 
                                 'end_x':receiver_coo[exp_receiver_dist][0], 'end_y':receiver_coo[exp_receiver_dist][1]}

        a = interception_coo - ball_coo
        b = receiver_coo - ball_coo    

        angle = np.arccos(np.clip(np.sum(a * b, axis=1) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1) + np.finfo(float).eps), -1, 1))
        exp_receiver_dist_and_angle = np.argmax((np.amin(dist) / dist) * (np.amin(angle) / angle))

        Intended_dict['dist and angle'] = {'ID':player_index_dict[exp_receiver_dist_and_angle], 
                                           'end_x':receiver_coo[exp_receiver_dist_and_angle][0], 'end_y':receiver_coo[exp_receiver_dist_and_angle][1]}

        if np.amin(angle) > 0.35:
            Intended_dict['dist and narrow angle'] = {'ID': None, 'end_x' : None, 'end_y' : None}
            events.at[idx,'Intended_Receiver'] = Intended_dict
            continue
        
        too_wide = np.where(angle > 0.35)[0]
        dist[too_wide] = np.inf

        exp_receiver_dist_and_angle_condition = np.argmax((np.amin(dist) / dist) * (np.amin(angle) / angle))

        Intended_dict['dist and narrow angle'] = {'ID': player_index_dict[exp_receiver_dist_and_angle_condition], 
                                           'end_x':receiver_coo[exp_receiver_dist_and_angle_condition][0], 'end_y':receiver_coo[exp_receiver_dist_and_angle_condition][1]}
        
        events.at[idx,'Intended_Receiver'] = Intended_dict
    return events

def preprocess_label(events):
    events = get_value_labels(events)
    events = get_intended_receiver(events)

    return events
    
