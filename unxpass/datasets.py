"""A PyTorch dataset containing all passes."""
import itertools
import os
from copy import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from torch.utils.data import Dataset
from src.preprocess_data import freeze_left_to_right

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


class PassesDataset(Dataset):
    def __init__(self, dataset, transform: Optional[Callable] = None):

        self.dataset = dataset
        self.dataset = self.actionfilter()
        self.transform = transform

    #패스 데이터 셋 & 헤딩패스는 제외(골킥패스는 포함) & freeze_frame이 존재하는것만
    def actionfilter(self):
        is_pass = self.dataset.eventName == 'Pass'
        #헤딩패스는 일단 포함
        # by_foot = ~dataset.subtype.str.contains("HEAD")
        by_foot = True
        in_freeze_frame = pd.notna(self.dataset.freeze_frame)

        return self.dataset[is_pass & by_foot & in_freeze_frame]

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict:
        if self.transform:
            field_dimen = (self.transform.x_bins, self.transform.y_bins)
            sample = self.play_left_to_right(self.dataset.iloc[idx], home_team_id='Home', field_dimen=field_dimen)
            sample = self.transform(sample)

        return sample
    
    #이벤트(패스)를 수행하는 팀을 왼쪽에 배치시킴
    def play_left_to_right(self, actions, home_team_id, field_dimen):
        ltr_actions = actions.copy()

        #away팀이 이벤트를 수행하고 있으면, 오른쪽에 있던 away을 flip시킴
        away_id = ltr_actions.team != home_team_id

        #이벤트데이터의 시작/끝위치 대칭
        if away_id:
            for col in ["start_x", "end_x"]:
                ltr_actions[col] = field_dimen[0] - ltr_actions[col]
            for col in ["start_y", "end_y"]:
                ltr_actions[col] = field_dimen[1] - ltr_actions[col]

            ltr_actions['freeze_frame'] = freeze_left_to_right(ltr_actions, field_dimen)
        return ltr_actions
        
class CompletedPassesDataset(PassesDataset): 
    def actionfilter(self, dataset: pd.DataFrame):
        # 부모 클래스의 필터링 조건을 적용
        filtered_dataset = super().actionfilter(dataset)
        # accurate=1인 데이터만 필터링
        return filtered_dataset[filtered_dataset.accurate == 1]

class FailedPassesDataset(PassesDataset):
    def actionfilter(self, dataset: pd.DataFrame):
        # 부모 클래스의 필터링 조건을 적용
        filtered_dataset = super().actionfilter(dataset)
        # accurate=0인 데이터만 필터링
        return filtered_dataset[filtered_dataset.accurate == 0]
