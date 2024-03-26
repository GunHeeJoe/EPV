import os, sys
import tqdm
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import pandas as pd
import ast
import matplotlib.pyplot as plt
import pickle
import torch
from src.visualization import plot_action
from src.preprocess_data import preprocess_data
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# this is very useful as it makes sure that always all columns and rows of a data frame are printed
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# datatools Animator
from datatools.metrica_helper import MetricaHelper
from datatools.trace_animator import TraceAnimator
from datatools.trace_helper import TraceHelper
import datatools.matplotsoccer as mps
from matplotlib import animation
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

match_list = ['match1', 'match2', 'match3']
all_events = pd.read_csv('../metrica-data/EPV-data/all-match.csv')

for match in match_list:
    track = pd.read_csv(f'../metrica-data/tracking-data/{match}.csv')
    track.set_index('frame',inplace=True)

    game_id = int(match[-1])
    event = all_events[(all_events['game_id'] == game_id) & (all_events['eventName'] == 'Pass')]

    for _, row in tqdm(event.iterrows()):
        pass_id = row['pass_id']
        session_id = row['session']
        start_frame = row['start_frame']
        end_frame = row['end_frame']

        session_traces = track[track['session'] == session_id]

        frame_traces = session_traces.loc[start_frame - 30 : end_frame + 30].reset_index()
        #animation은 t가 0부터시작하기 때문에, tracking-data의 frame, time정보를 활용할 수 없음 -> 따로 time_index제작함
        highlight_time = frame_traces[(frame_traces['frame'] >= start_frame) & (frame_traces['frame'] <= end_frame)].index.to_list()

        animator = TraceAnimator(
            trace_dict={"main": frame_traces},
            highlight_time = highlight_time,
            show_episodes=True,
            show_events=True,
            play_speed=1,
        )
        
        anim = animator.run()
        anim_file = f'./pass-animation/{pass_id}.mp4'
        writer = animation.FFMpegWriter(fps=10)
        anim.save(anim_file, writer=writer)