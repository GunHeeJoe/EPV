import os, sys
import tqdm
import time
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt


def extract_pass(df_event, df_track, method=1):
    # get all passes
    df_pass = df_event[
                    (df_event["eventName"] == "Pass")
                    & (df_event["start_frame"] < df_event["end_frame"])
                ].copy()


    # only keep relevant columns
    df_pass = df_pass[["team", "start_frame", "end_frame", "eventName", "subEventName", "from", "to", "freeze_frame" ,"accurate", "pass_id"]].copy()

    # attach the ball position at the start and end frame
    df_ball = df_track[["frame", "ball_x", "ball_y", "episode"]].copy()

    df_ball.columns = ["start_frame", "xPosStart", "yPosStart", "episodeStart", "ballInPlayStart"]
    df_pass = pd.merge(df_pass, df_ball, how="left")

    df_ball.columns = ["end_frame", "xPosEnd", "yPosEnd", "episodeEnd", "ballInPlayEnd"]
    df_pass = pd.merge(df_pass, df_ball, how="left")
    df_pass = df_pass[df_pass["xPosEnd"].notnull()].copy()

    # only keep passes for which the ball was in play at the beginning of the pass (i.e. exclude throw-ins)
    df_pass = df_pass[df_pass["ballInPlayStart"] == 1].copy()
    df_pass.reset_index(inplace=True, drop=True)

    # extract accurate or not by method
    df_pass = df_pass[df_pass["accurate"] == method].copy()

    return df_pass



def extract_player_pos(df, frame):
   # metrica data에서 선수들 위치 추출하는 데이터 프레임
   basic = df[df['frame'] == frame]

   # player 전처리
   player_df = []
   for i in range(1, 29):
     if i < 10:
       i_basic = [i]
       i_basic += list(basic.filter(regex=f'0{i}').iloc[0])[0:2]
       i_basic += ['Home']
     else:
       i_basic = [i]
       i_basic += list(basic.filter(regex=f'{i}').iloc[0])[0:2]
       if i <= 14 :
         i_basic += ['Home']
       else:
         i_basic += ['Away']
     player_df.append(i_basic)
   player_df = pd.DataFrame(player_df, columns=['playerId', 'xPos', 'yPos', 'team']).dropna(axis=0)
   
   return player_df


def intended_receiver(dataframe):
    # intended receiver를 구합니다.
    # 거리, 거리/각도, 거리/각도(narrow만 고려) 기반 모두 구합니다.
    intended_list = []
    frame_dist = []
    frame_receiver = []
    frame_ball = []
    frame_angle = []


    return final_df