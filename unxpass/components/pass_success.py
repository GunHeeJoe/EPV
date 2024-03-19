"""Implements the pass success probability component."""
from typing import Any, Dict, List
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
from rich.progress import track
"""Model architectures."""
from typing import Callable, Dict, List, Optional, Union
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import random
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from unxpass.datasets import PassesDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from collections import defaultdict
#from unxpass.sampler import SubsetRandomSampler_function, WeightedRandomSampler_function,My_Sampler

from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .soccermap import SoccerMap, pixel
from torchmetrics.classification import BinaryCalibrationError


class PassSuccessComponent():
    """The pass success probability component.

    From any given game situation where a player controls the ball, the model
    estimates the success probability of a pass attempted towards a potential
    destination location.
    """

    component_name = "pass_success"

    def __init__(self, model):
        
        self.model = model.to('cuda:0')
        logger = TensorBoardLogger("./runs/", name="pass_success")
        checkpoint_callback = ModelCheckpoint(
            dirpath=logger.log_dir,
            filename='{val_loss:.2f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )

        # 조기 종료 콜백 설정 (예시로 추가함)
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            strict=False,
            min_delta=1e-3,
            verbose=True,
            mode='min'
        )

        # Trainer 설정
        self.trainer = pl.Trainer(
            max_epochs=30,
            callbacks=[checkpoint_callback, early_stop_callback],  # 여기에 콜백 리스트 추가
            logger=logger,
            accelerator="cuda",
            devices=1,
            strategy='auto'
        )

        # self.trainer = pl.Trainer(max_epochs=1,callbacks=early_stop_callback,
        #                      logger=logger,
        #                      accelerator="cuda",devices=1, strategy='auto', checkpoint_callback=checkpoint_callback)
        # self.trainer = pl.Trainer(max_epochs=1,callbacks=early_stop_callback,
        #                      logger=logger)
        
    def train(
        self,
        train_dataset,
        valid_dataset,
        optimized_metric=None,
        callbacks=None,
        logger=None,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        **train_cfg,
    ) -> Optional[float]:
        #mlflow.pytorch.autolog()
        #mlflow.pytorch.autolog(disable=True)
        # Init lightning trainer

        #trainer = pl.Trainer(callbacks=callbacks, logger=logger, **train_cfg["trainer"])
        
        #EarlyStopping parameter
        #monitor='val_loss',  # 모니터링할 지표: 검증 손실
        #min_delta=0,         # 최소 변화량 (기본값 0)
        #patience=3,          # 성능 개선이 없을 때 허용할 에포크 수 (기본값 3)
        #mode='min',          # 손실 값이 감소하는 방향으로 개선 여부를 판단
        #strict=True,         # 지정된 지표보다 성능이 개선되어야 조기 종료 허용
        #check_finite=True,   # 손실 값이 유효한지 확인
        #stop_on_nan=False,   # 손실 값이 NaN인 경우 훈련 종료하지 않음
        #verbose=False,       # 조기 종료 시 메시지를 표시하지 않음
        #check_interval='epoch'  # 에포크마다 조기 종료 확인
        # early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=1e-3,
        #                                     mode='min',patience=3)
        # trainer = pl.Trainer(max_epochs=1,callbacks=early_stop_callback,
        #                      logger=self.logger,
        #                      accelerator="cuda", devices=2, strategy="ddp")
    

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Print path to best checkpoint
        #log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            return self.trainer.callback_metrics[optimized_metric]

        return None

     #samples : predict 
    #true_labels : target
    def expected_calibration_error(self, samples, true_labels, M=10):
        # uniform binning approach with M number of bins
        #(0, 0.1), (0.1, 0.2), (0.2, 0.3)......(0.9, 1.0)의 (lower, upper)쌍이 나옴
        bin_boundaries = np.linspace(0, 1, M + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # keep confidences / predicted "probabilities" as they are
        confidences = samples
        # get binary class predictions from confidences
        predicted_label = (samples>0.5).astype(float)

        # get a boolean list of correct/false predictions
        accuracies = predicted_label==true_labels
        positive_class = true_labels==1

        soccermap_ece = np.zeros(1)
        epv_ece = np.zeros(1)
        soccermap_ece1 = np.zeros(1)
        epv_ece1 = np.zeros(1)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            # confidence가 지정된 값 사이의 존재하는지 여뷰 
            in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
            
            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            # 전체 데이터 중 해당 Bin에 해당하는 확류ㅠㄹ
            prop_in_bin = in_bin.astype(float).mean()
            
            if prop_in_bin.item() > 0:
                # get the accuracy of bin m: acc(Bm)
                #accuracy
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                #predict
                label_in_bin = predicted_label[in_bin].astype(float).mean()

                #둘다 confidence=예측값을 활용하고, 추가로
                #epv'ece는 true_label의 평균을 활용하고
                ground_truth = true_labels[in_bin].astype(float).mean()
                #soccermap'ece는 (true_label=1)의 평균을 활용함
                positive_class_in_bin = positive_class[in_bin].astype(float).mean()

                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = confidences[in_bin].mean()

                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                # soccermap_ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                # epv_ece += np.abs(avg_confidence_in_bin - label_in_bin) * prop_in_bin

                soccermap_ece1 += np.abs(avg_confidence_in_bin - positive_class_in_bin) * prop_in_bin
                epv_ece1 += np.abs(avg_confidence_in_bin - ground_truth) * prop_in_bin
                #print(f"{(bin_lower, bin_upper)} -> {(avg_confidence_in_bin,accuracy_in_bin)} => {(avg_confidence_in_bin-accuracy_in_bin)}")
            
        # print(f"original used evalution : soccermap'ece = {soccermap_ece}, epv'ece = {epv_ece}")
        # print(f"new used evalution : soccermap'ece1 = {soccermap_ece1}, epv'ece1 = {epv_ece1}")
        return soccermap_ece1, epv_ece1

    def _get_metrics(self, y, y_hat):
        bce_l1 = BinaryCalibrationError(n_bins=10,norm='l1')
        tensor_y_hat =  torch.Tensor(y_hat)
        tensor_y =  torch.Tensor(y)

        soccermap_ece, epv_ece = self.expected_calibration_error(y_hat,y)
        y_pred = y_hat > 0.5
        return {
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "log_loss": log_loss(y, y_hat),
            "brier": brier_score_loss(y, y_hat),
            "roc_auc": roc_auc_score(y, y_hat),
            "soccermap_ece" : soccermap_ece,
            "epv_ece" : epv_ece,
            "bce_l1" : bce_l1(tensor_y_hat, tensor_y),
        }

    def test(self, test_dataset, batch_size=1, num_workers=0, pin_memory=True, **test_cfg) -> Dict[str, float]:
      
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
 
        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()
        self.model = self.model.to('cpu')
        # Apply model on test set
        all_preds, all_targets = [], []
        # for batch in track(dataloader,description="pass success model testing"):
        for batch in track(dataloader,description="pass success model testing"):
            loss, y_hat, y = self.model.step(batch)
            all_preds.append(y_hat)
            all_targets.append(y)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]
        all_targets = torch.cat(all_targets, dim=0).detach().numpy()[:, 0]

        # Compute metrics
        return self._get_metrics(all_targets, all_preds)

    def predict(self, test_dataset, batch_size=1, num_workers=0, pin_memory=False) -> pd.Series:
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()

        # Apply model on test set
        all_preds = []
        for batch in track(dataloader):
            loss, y_hat, y = self.model.step(batch)
            all_preds.append(y_hat)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]

        return pd.Series(all_preds, index=test_dataset.dataset.index)


    def predict_surface(self, test_dataset, batch_size=1, num_workers=0, pin_memory=False, **predict_cfg) -> Dict:
        actions = test_dataset.dataset.reset_index()
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        #predict_surface는 인덱스에 맞는 예측값을 output에 넣는거라서 
        #gpu 2개 써버리면 index가 안맞는 형식으로 출력이 되버림
        predictor = pl.Trainer(**predict_cfg.get("trainer", {}),devices=1)
        predictions = torch.cat(predictor.predict(self.model, dataloaders=dataloader))
        self.trainer.device=1

        output = defaultdict()
        for i, row in actions.iterrows():
            # predictions : data x 1 x 72 x 108이라서 i번째 데이터를 불러오는 방법predictions[i][0]
            output[row['pass_id']] = predictions[i][0].detach().numpy()

        return output

#nn.Module처럼 모델 내부의 구조를 설계
class PytorchSoccerMapModel(pl.LightningModule):
    """A pass success probability model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr : float = 1e-5
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        #self.save_hyperparameters()

        self.model = SoccerMap(in_channels=13)

        self.sigmoid = nn.Sigmoid()

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)

        x = self.sigmoid(x)

        return x

    def step(self, batch: Any):
        x, mask, y = batch

        surface = self.forward(x)
    
        #해당 목적지의 단일 예측값만 loss로 사용

        y_hat = pixel(surface, mask)
        
        #처음에, intended+end 둘의 pixel값을 더한 값으로 backpropagation하려고 했는데, 
        #생각해보니 둘을 더하면 predict>1를 넘어가버려서 error가 발생함
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        #print(torch.unique(targets,return_counts=True))
        
        # log train metrics
        # name : 로그이름
        # value : 기록할 값
        # on_step : 훈련스탭마다 로그 기록
        # prog_bar : 진행 막대 표시
        # logger : 로그 기록을 위한 looger
        # sync_dist : 분산 환경에서 로그 동기화
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #self.logger.experiment.add_scalar("train/loss",loss)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #self.logger.experiment.add_scalar("val/loss",loss)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        #self.log("loss/test", loss, on_step=False, on_epoch=True,prog_bar=True,sync_dist='auto')
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, _, _ = batch
        surface = self(x)
        return surface

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        return torch.optim.Adam(params=self.parameters(), lr=1e-4)
    
class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim

    def _get_cell_indexes(self, x, y):
        #FIFA에서 선정한 축구장의 규격이 68x105라서 스케일을 맞추 코드
        # x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        # y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)

        #Metrica-data는 72x108이 규격이므로 스케일을 할 필요는 없음
        x_bin = np.clip(x, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        #frame = pd.DataFrame.from_records(sample["freeze_frame"],orient='index')
        frame = pd.DataFrame.from_dict(sample["freeze_frame"],orient='index')
        #success model은 도착한 패스의 pixel부분의 성공여부로 학습함
        target = int(sample["accurate"])

        ball_coo = frame.loc[frame.ball,['start_x', 'start_y']].values.reshape(-1, 2)

        goal_coo = np.array([[108, 36]])

        # Output
        matrix = np.zeros((13, self.y_bins, self.x_bins))

        # Locations of the passing player's teammates
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["start_x", "start_y"]].values.reshape(-1, 2)
        players_att_vx  = frame.loc[~frame.actor & frame.teammate, "start_vx"].values
        players_att_vy  = frame.loc[~frame.actor & frame.teammate, "start_vy"].values

        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.teammate & ~frame.ball, ["start_x", "start_y"]].values.reshape(-1, 2)
        players_def_vx  = frame.loc[~frame.teammate & ~frame.ball, "start_vx"].values
        players_def_vy  = frame.loc[~frame.teammate & ~frame.ball, "start_vy"].values

        
        # 현재 상태의 위치정보
        # CH 1: Locations of attacking team
        # 공격팀의 위치정보
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )

        matrix[0, y_bin_att, x_bin_att] = 1
        matrix[1, y_bin_att, x_bin_att] = players_att_vx
        matrix[2, y_bin_att, x_bin_att] = players_att_vy

        # CH 2: Locations of defending team
        # 수비팀의 위치정보
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        # for i in range(len(x_bin_def)):
        #     matrix[0, y_bin_def[i], x_bin_def[i]] += 1
        matrix[3, y_bin_def, x_bin_def] = 1
        matrix[4, y_bin_def, x_bin_def] = players_def_vx
        matrix[5, y_bin_def, x_bin_def] = players_def_vy

        # np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins] : 0.5~108(74)사이를 1간격 = [0.5, 1.5, 2.5.....107.5] & [0.5, 1.5, 2.5, .....73.5] 
        # 각 pixel별 중앙위치값 : (0,0)의 셀의 위치좌표는 (0.5,0.5)
        # 셀과 공/골대/위치사이의 거리및 각도를 구할 때 주의사항 : 공/골대/위치에 매핑되는 인덱스+0.5를 수행해줘야함
        # 셀은 해당 셀의 중앙위치를 좌표로 설정하지만, 인덱스는 get_cell_index에서 Bottom-left에 설정하므로 같은 셀간의 거리가 [0.5, 0.5]로 계산되는 문제가 발생함
        # 즉, 선수의 위치(107.4,36.5)에 매핑되는 셀은 (107,36)이지만, other-location은 중앙위치인 (107.5, 36.5)이므로 같은 셀간의 거리가 (0.5, 0.5)차이나는 문제가 발생함  
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        # CH 7: Distance to goal
        # 모든 pixel에서 골대까지 거리
        x_goal, y_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        #(x_goal+0.5, y_goal+0.5)해야하는 이유 : other-location은 셀의 중앙값으로 위치를 설정하지만, get_cell_index는 bottom-left의 위치로 설정하기 때문이다.
        #만약 0.5를 안 더한다면? : 같은 셀간의 거리가 sqrt(0.5**2 + 0.5**2)=0.7이 나오는 문제가 발생함
        #matrix[6, :, :] = np.sqrt((x_goal - xx) ** 2 + (y_goal - yy) ** 2)
        matrix[6, :, :] = np.sqrt(((x_goal+0.5) - xx) ** 2 + ((y_goal+0.5) - yy) ** 2)

        # CH 8: Distance to event-player
        # 이벤트를 수행하는 선수의 위치과 셀간의 거리가 아님
        # 이벤트를 수행하는 선수의 위치에 해당하는 셀과 다른 셀간의 거리 -> 애초에 SoccerMap Architecture는 모든 feature들이 cell단위로 계산되기 때문이다.
        event_player_coo = frame.loc[frame.actor, ["start_x", "start_y"]].values.reshape(-1, 2)
        x_bin_event_player, y_bin_event_player = self._get_cell_indexes(event_player_coo[:, 0], event_player_coo[:, 1],)
        #0.5를 안 더해주면, CH7처럼 같은 셀간의 거리가 0.7이 나오는 문제 발생함
        #matrix[7, :, :] = np.sqrt((xx - x_bin_event_player) ** 2 + (yy - y_bin_event_player) ** 2)
        matrix[7, :, :] = np.sqrt((xx - (x_bin_event_player+0.5)) ** 2 + (yy - (y_bin_event_player+0.5)) ** 2)

        # CH 9 : Angle to the goal location
        # 0.5를 안 더해주면, x,y의 차이가 각각 0.5씩 차이나므로 tan(0.5/0.5)=tan(1)=45(degree)=0.78539816(radian)이 나오는 문제가 발생함
        #matrix[8, :, :] = np.abs(np.arctan((y_goal - yy) / (x_goal - xx)))
        #np.abs(np.arctan(y/x)) : radian을 구하는 작업으로 가로기준으로 대칭되는 위/아래 셀은 골대까지의 각도가 양수/음수가 되지 않도록 절대값을 취함
        #np.finfo(float).eps : 골대위치에 해당하는 셀과 평행한 셀과의 각도는 0도이므로 NAN값이 나옴
        matrix[8, :, :] = np.abs(np.arctan(((y_goal+0.5) - yy) / ((x_goal+0.5) - xx + np.finfo(float).eps)))

        # CH 10 : Sine of the angle between possession team player's and the ball location
        # CH 11 : Cos of the angle between possession team player's and the ball location
        # 공격팀의 위치과 공의 위치간의 거리가 아님
        # 공격팀의 위치에 해당하는 셀과 공의 위치에 해당하는 셀간의 거리임
        # 선수의 위치 / 공의 위치 모두 get_cell_index(bottom-left)에서 불러왔으므로 0.5를 더해줄 필요는 없음(그냥 했음..)
        #np.finfo(float).eps : 선수의 위치에 해당하는 셀과 공의 위치에 해당하는 셀이 평행하면, 각도는 0도이므로 NAN값이 나옴
        x_ball, y_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        radian_between_attacking_ball = np.abs(np.arctan(((y_ball+0.5) - (y_bin_att+0.5)) / ((x_ball+0.5) - (x_bin_att+0.5) + np.finfo(float).eps)))

        matrix[9, y_bin_att, x_bin_att] = np.sin(radian_between_attacking_ball)
        matrix[10, y_bin_att, x_bin_att] = np.cos(radian_between_attacking_ball)

        # CH 12 : Sine of the angle between the ball carrier's velocity and every other location
        # CH 13 : Cosine of the angle between the ball carrier;s velocity and every other location
        # 볼 소유한 선수의 속도 벡터과 볼 소유한 선수에 해당하는 셀과 다른 셀를 잇는 벡터 사이의 각도 -> cell단위로 추출 및 계산되는점에 주의!
        # 0.5를 안 더해주면, CH7처럼 같은 셀간 벡터가 (0.5, 0.5)가 나오는 문제가 발생함
        # np.finfo(float).eps : 이벤트를 수행하는 선수의 위치에 해당하는 셀이면, vec_event_player_cell가 영벡터
        # 만약 선수가 cell A위치로 향하고 있다면, 각도는 0도 -> sin(0)=0, cos(0)=1                   
        event_player_velocity = frame.loc[frame.actor, ["start_vx", "start_vy"]].values.reshape(-1, 2)
        vec_event_player_cell = np.dstack(np.meshgrid(xx - (x_bin_event_player+0.5), yy - (y_bin_event_player+0.5)))
        cosine_between_carrier_attacking = np.clip(
            np.sum(event_player_velocity * vec_event_player_cell, axis=2) /  (np.linalg.norm(event_player_velocity,axis=1) * np.linalg.norm(vec_event_player_cell,axis=2) + np.finfo(float).eps), -1, 1
        )
        matrix[11, :, :] = np.sqrt(1 - cosine_between_carrier_attacking ** 2)
        matrix[12, :, :] = cosine_between_carrier_attacking

        # Mask
        #패스 도착지에 확률값을 바탕으로 backpropagation
        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_ball_coo = frame.loc[frame.ball,['end_x', 'end_x']].values.reshape(-1,2)

        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        
        x_ball_end, y_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y_ball_end, x_ball_end] = 1

        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            torch.tensor([target]).float(),
        )

