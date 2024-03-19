#1. 모든 액션의 가치를 레이블링해야한다
#2. 소유권이란 session이 바뀌거나 득점이 된 경우를 의미한다
#3. 득점/실점을 할 경우 이전 소유권(session의 시작 / 이전 득실점)과 현재 골 사이의 모든 행동들은 Attacking_team=1, defending_team=-1를 받는다
#4. 시간 앱실론 = 15초 : 단기적 가치과 장기적 가치의 균형을 마치기 위해서 사용되는 시간상수로 현재 골 이전의 15초동안 발생한 행동만 보상을 받는다
#5. 우리는 세트피스동안 발생하는 행동(프리킥, 코너킥, 페널티킥등)은 분석에서 제외한다 -> type=SET PIECE인 행동

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