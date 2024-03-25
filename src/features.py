import os, sys
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt


# 성공 혹은 실패한 패스를 추출하는 함수
def extract_pass(df_event, df_track, method=1):
    # get all passes
    df_pass = df_event[
                    (df_event["eventName"] == "Pass")
                    & (df_event["start_frame"] < df_event["end_frame"])
                ].copy()

    # attach the ball position at the start and end frame
    df_ball = df_track[["frame", "ball_x", "ball_y", "episode"]].copy()

    df_ball.columns = ["start_frame", "xPosStart", "yPosStart", "episodeStart"]
    df_pass = pd.merge(df_pass, df_ball, how="left")

    df_ball.columns = ["end_frame", "xPosEnd", "yPosEnd", "episodeEnd"]
    df_pass = pd.merge(df_pass, df_ball, how="left")
    df_pass = df_pass[df_pass["xPosEnd"].notnull()].copy()

    # only keep passes for which the ball was in play at the beginning of the pass (i.e. exclude throw-ins)
    df_pass = df_pass[df_pass["episodeStart"] != 0].copy()
    df_pass.reset_index(inplace=True, drop=True)

    # extract accurate or not by method
    df_pass = df_pass[df_pass["accurate"] == method].copy()

    return df_pass



# metrica data에서 선수들 위치 추출하는 데이터 프레임
def extract_player_pos(df, frame):
   basic = df[df['frame'] == frame]
   # x, y 칼럼만 뽑아내기
   columns = []
   c_player = []
   for i in list(basic.columns):
       if (i[3:5] == '_x') or (i[3:5] == '_y'):
           columns.append(i)
           c_player.append(i[0:3])
   c_player = list(set(c_player))
   basic = basic[columns]

   # player 전처리
   player_df = []
   for i in c_player:
       i_basic = []
       i_basic += [i]
       i_basic += [basic.filter(regex=f'{i}_x').iloc[0][0], basic.filter(regex=f'{i}_y').iloc[0][0]]
       if i[0] == 'A':
          i_basic += ['Home']
       else:
          i_basic += ['Away']
       player_df.append(i_basic)
   player_df = pd.DataFrame(player_df, columns=['playerId', 'xPos', 'yPos', 'team']).dropna(axis=0)
   
   # 마지막에 A/B로 sort 하고, 그 다음에 숫자로 sort
   player_df = player_df.sort_values(by=['playerId'][0], axis = 0)
   
   return player_df



# intended receiver를 구합니다.
def intended_receiver(df_pass, df_track):
    # 거리, 거리/각도, 거리/각도(narrow만 고려)
    intended_list = []

    for idx, action in tqdm(df_pass.iterrows()):
      # 선수들의 좌표 가져오기
      df_start_frame = extract_player_pos(df_track, action.start_frame)
      id_list = list(df_start_frame["playerId"])

      receiver_coo = np.array(
          [
              (o["xPos"], o["yPos"])
              for index, o in df_start_frame.iterrows()
          ]
      )

      # 패스한 선수는 제거
      passer_index = id_list.index(action['from'])
      del id_list[passer_index]
      receiver_coo = np.delete(receiver_coo, passer_index, axis = 0)

      # Home, Away 팀만 추출
      len_home = len(df_start_frame[df_start_frame['team'] == 'Home'])
      if action['team'] == 'Home':
        receiver_coo = receiver_coo[:len_home]
        id = id_list[:len_home]
      else:
        receiver_coo = receiver_coo[len_home:]
        id = id_list[len_home:]

      # 패스의 시작, 끝 위치 가져오기
      ball_coo = np.array([action.xPosStart, action.yPosStart])
      interception_coo = np.array([action.xPosEnd, action.yPosEnd])

      # 볼의 마지막 위치와 선수들의 위치 간의 거리 계산
      # 거리 기반 계산 : Baseline 1 
      dist = np.sqrt(
                    (receiver_coo[:, 0] - interception_coo[0]) ** 2
                    + (receiver_coo[:, 1] - interception_coo[1]) ** 2
                )

      # 패스 라인과 선수들 간의 각도 계산
      # 거리 & 각도 기반 계산 : Baseline 2
      a = interception_coo - ball_coo
      b = receiver_coo - ball_coo
      angle = np.arccos(
                    np.clip(
                        np.sum(a * b, axis=1) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)), -1, 1
                    )
                )
      
      # 거리 & 각도(20도 제한) : Baseline 3
      too_wide = np.where(angle > 0.35)[0]
      dist_2 = dist.copy()
      dist_2[too_wide] = np.inf

      # intended receiver 찾기
      # TODO: you could play around with the weight given to the distance and angle here
      exp_receiver_1 = np.argmax((np.amin(dist) / dist) * 1)
      exp_receiver_2 = np.argmax((np.amin(dist) / dist) * (np.amin(angle) / angle))
      exp_receiver_3 = np.argmax((np.amin(dist_2) /dist_2) * (np.amin(angle) / angle))

      # 계산값 dictionary로 정리
      intended_dic = {}
      c = 0
      for i in [exp_receiver_1, exp_receiver_2, exp_receiver_3]:
        dic = {}
        dic["ID"] = id[i]
        dic["end_x"] = receiver_coo[i][0]
        dic["end_y"] = receiver_coo[i][1]
        if c == 0:
          intended_dic["dist"] = dic
        elif c == 1:
          intended_dic["dist and angle"] = dic
        else:
          intended_dic["dist and narrow angle"] = dic
        c += 1

      
      intended_list.append(intended_dic)

    df_pass['Intended_Receiver'] = intended_list
    
    return df_pass