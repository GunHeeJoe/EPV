"""Implements the SoccerMap architecture."""
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.preprocessing import StandardScaler

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)

        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_block, in_channel, init_weights=True):
        super().__init__()

        self.in_channels=64 
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 1)
        self.conv5_x = self._make_layer(block, 64, num_block[3], 1)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18(in_channel):
    return ResNet(BasicBlock, [2,2,2,2],in_channel)

def resnet34(in_channel):
    return ResNet(BasicBlock, [3, 4, 6, 3],in_channel)

def resnet50(in_channel):
    return ResNet(BottleNeck, [3,4,6,3],in_channel)

def resnet101(in_channel):
    return ResNet(BottleNeck, [3, 4, 23, 3],in_channel)

def resnet152(in_channel):
    return ResNet(BottleNeck, [3, 8, 36, 3],in_channel)

class _FeatureExtractionLayer(nn.Module):
    """The 2D-convolutional feature extraction layer of the SoccerMap architecture.

    The probability at a single location is influenced by the information we
    have of nearby pixels. Therefore, convolutional filters are used for
    spatial feature extraction.

    Two layers of 2D convolutional filters with a 5 × 5 receptive field and
    stride of 1 are applied, each one followed by a ReLu activation function.
    To keep the same dimensions after the convolutional filters, symmetric
    padding is applied. It fills the padding cells with values that are
    similar to those around it, thus avoiding border-image artifacts that can
    hinder the model’s predicting ability and visual representation.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=(5, 5), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding="valid")
        # (left, right, top, bottom)
        self.symmetric_padding = nn.ReplicationPad2d((2, 2, 2, 2))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)

        return x

class ResNet_FeatureExtractionLayer(nn.Module):
    """The 2D-convolutional feature extraction layer of the SoccerMap architecture.

    The probability at a single location is influenced by the information we
    have of nearby pixels. Therefore, convolutional filters are used for
    spatial feature extraction.

    Two layers of 2D convolutional filters with a 5 × 5 receptive field and
    stride of 1 are applied, each one followed by a ReLu activation function.
    To keep the same dimensions after the convolutional filters, symmetric
    padding is applied. It fills the padding cells with values that are
    similar to those around it, thus avoiding border-image artifacts that can
    hinder the model’s predicting ability and visual representation.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, 128, kernel_size=(5, 5), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(5, 5), stride=1, padding="valid")
        # (left, right, top, bottom)
        self.symmetric_padding = nn.ReplicationPad2d((2, 2, 2, 2))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)

        return x
    
class _PredictionLayer(nn.Module):
    """The prediction layer of the SoccerMap architecture.

    The prediction layer consists of a stack of two convolutional layers, the
    first with 32 1x1 convolutional filters followed by an ReLu activation
    layer, and the second consists of one 1x1 convolutional filter followed by
    a linear activation layer. The spatial dimensions are kept at each step
    and 1x1 convolutions are used to produce predictions at each location.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)  # linear activation
        return x


class _UpSamplingLayer(nn.Module):
    """The upsampling layer of the SoccerMap architecture.

    The upsampling layer provides non-linear upsampling by first applying a 2x
    nearest neighbor upsampling and then two layers of convolutional filters.
    The first convolutional layer consists of 32 filters with a 3x3 activation
    field and stride 1, followed by a ReLu activation layer. The second layer
    consists of 1 layer with a 3x3 activation field and stride 1, followed by
    a linear activation layer. This upsampling strategy has been shown to
    provide smoother outputs.
    """

    def __init__(self):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding="valid")
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = self.symmetric_padding(x)
        x = self.conv2(x)  # linear activation
        x = self.symmetric_padding(x)
        return x

class ResNet_UpSamplingLayer(nn.Module):
    """The upsampling layer of the SoccerMap architecture.

    The upsampling layer provides non-linear upsampling by first applying a 2x
    nearest neighbor upsampling and then two layers of convolutional filters.
    The first convolutional layer consists of 32 filters with a 3x3 activation
    field and stride 1, followed by a ReLu activation layer. The second layer
    consists of 1 layer with a 3x3 activation field and stride 1, followed by
    a linear activation layer. This upsampling strategy has been shown to
    provide smoother outputs.
    """

    def __init__(self, in_channel):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding="valid")
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = self.symmetric_padding(x)
        x = self.conv2(x)  # linear activation
        x = self.symmetric_padding(x)
        return x
    
class _FusionLayer(nn.Module):
    """The fusion layer of the SoccerMap architecture.

    The fusion layer merges the final prediction surfaces at different scales
    to produce a final prediction. It concatenates the pair of matrices and
    passes them through a convolutional layer of one 1x1 filter.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1)

    def forward(self, x: List[torch.Tensor]):
        out = self.conv(torch.cat(x, dim=1))  # linear activation
        return out


class SoccerMap(nn.Module):
    """SoccerMap architecture.

    SoccerMap is a deep learning architecture that is capable of estimating
    full probability surfaces for pass probability, pass slection likelihood
    and pass expected values from spatiotemporal data.

    The input consists of a stack of c matrices of size lxh, each representing a
    subset of the available spatiotemporal information in the current
    gamestate. The specific choice of information for each of these c slices
    might vary depending on the problem being solved

    Parameters
    ----------
    in_channels : int, default: 13
        The number of spatiotemporal input channels.

    References
    ----------
    .. [1] Fernández, Javier, and Luke Bornn. "Soccermap: A deep learning
       architecture for visually-interpretable analysis in soccer." Joint
       European Conference on Machine Learning and Knowledge Discovery in
       Databases. Springer, Cham, 2020.
    """

    def __init__(self, in_channels=15):
        super().__init__()

        # Convolutions for feature extraction at 1x, 1/2x and 1/4x scale
        self.features_x1 = _FeatureExtractionLayer(in_channels)
        self.features_x2 = _FeatureExtractionLayer(64)
        self.features_x4 = _FeatureExtractionLayer(64)  
        self.ResNet_FeatureExtractionLayer = ResNet_FeatureExtractionLayer(in_channels=256)

        # Layers for down and upscaling and merging scales
        self.up_x2 = _UpSamplingLayer()
        self.up_x4 = _UpSamplingLayer()
        self.up = ResNet_UpSamplingLayer(in_channel=256)
        self.down_x2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fusion_x2_x4 = _FusionLayer()
        self.fusion_x1_x2 = _FusionLayer()

        # Prediction layers at each scale
        self.prediction_x1 = _PredictionLayer()
        self.prediction_x2 = _PredictionLayer()
        self.prediction_x4 = _PredictionLayer()

        self.resnet15 = resnet50(in_channel=15)
        self.resnet64 = resnet50(in_channel=64)
        self.resnet256 = resnet50(in_channel=256)

    def forward(self, x):
        # Feature extraction
        f_x1 = self.features_x1(x)
        f_x2 = self.up(self.resnet15(x))
        f_x4 = self.ResNet_FeatureExtractionLayer(self.resnet15(x))

        # Prediction
        pred_x1 = self.prediction_x1(f_x1)
        pred_x2 = self.prediction_x2(f_x2)
        pred_x4 = self.prediction_x4(f_x4)

        # Fusion
        x4x2combined = self.fusion_x2_x4([self.up_x4(pred_x4), pred_x2])
        combined = self.fusion_x1_x2([self.up_x2(x4x2combined), pred_x1])

        # The activation function depends on the problem
        return combined


def pixel(surface, mask):
    """Return the prediction at a single pixel.

    This custom layer is used to evaluate the loss at the pass destination.

    Parameters
    ----------
    surface : torch.Tensor
        The final prediction surface.
    mask : torch.Tensor
        A sparse spatial representation of the final pass destination.

    Returns
    -------
    torch.Tensor
        The prediction at the cell on the surface that matches the actual
        pass destination.
    """
    # 어차피 mask는 해당 목적지의 값만 1
    # surface는 모든 표면의 확률 값이 분포되어있는데,
    # 둘의 pixel끼리 곱한 후 전체 합 = 목적지의 surface값만 추출됨
    masked = surface * mask
    value = torch.sum(masked, dim=(3, 2))
    return value


class soccermap_ToSoccerMapTensor:
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
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        start_x, start_y, end_x, end_y = (
            sample["start_x_a0"],
            sample["start_y_a0"],
            sample["end_x_a0"],
            sample["end_y_a0"],
        )
   
        intended_end_x, intended_end_y = (
            sample["intended_end_x_a0"],
            sample["intended_end_y_a0"],
        )   
        ball_height_ground, ball_height_low, ball_height_high = (
            sample["ball_height_ground_a0"],
            sample["ball_height_low_a0"],
            sample["ball_height_high_a0"]
        )
        #intended_end_x, intended_end_y = (sample['intended_end_x_a0'], sample['intended_end_y_a0'])
        scaler = StandardScaler()
        speed_x, speed_y = sample["speedx_a02"], sample["speedy_a02"]
        frame = pd.DataFrame.from_records(sample["freeze_frame_360_a0"])
        target = int(sample["success"]) if "success" in sample else None

        # Location of the player that passes the ball
        # passer_coo = frame.loc[frame.actor, ["x", "y"]].fillna(1e-10).values.reshape(-1, 2)
        # Location of the ball
        ball_coo = np.array([[start_x, start_y]])
        ball_end_coo = np.array([[end_x, end_y]])
        # Location of the goal
        goal_coo = np.array([[105, 34]])

        # Output
        matrix = np.zeros((15, self.y_bins, self.x_bins))

        # Locations of the passing player's teammates
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["x", "y"]].values.reshape(-1, 2)
        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

        # 현재 상태의 위치정보
        # CH 1: Locations of attacking team
        # 공격팀의 위치정보
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        # 수비팀의 위치정보
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1
        
        # CH 3: Distance to ball
        # 모든 pixel에서 시작공까지 거리
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        # CH 4: Distance to goal
        # 모든 pixel에서 골대까지 거리
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        # CH 5: Cosine of the angle between the ball and goal
        # 모든 위치에서 시작공과 골대사이의 코사인값
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 6: Sine of the angle between the ball and goal
        # sin = np.cross(a,b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
        # 모든 위치에서 시작공과 골대사이의 사인값
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)  # This is much faster

        # CH 7: Angle (in radians) to the goal location
        # 모든 위치에서 골대까지 각도
        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        # CH 8-9: Ball speed
        if speed_x<0 or speed_x>50:
            speed_x = 0
        if speed_y<0 or speed_y>50:
            speed_x = 0 
        # matrix[7, y0_ball, x0_ball] = speed_x
        # matrix[8, y0_ball, x0_ball] = speed_y   
        matrix[7, :, :] = speed_x
        matrix[8, :, :] = speed_y

        # CH 9: Distance to ball_end
        # 모든 pixel에서 도착지까지 거리
        x0_ball_end, y0_ball_end = self._get_cell_indexes(ball_end_coo[:, 0], ball_end_coo[:, 1])
        matrix[9, :, :] = np.sqrt((xx - x0_ball_end) ** 2 + (yy - y0_ball_end) ** 2)

        # CH 10: Cosine of the angle between the ball_end and goal
        # 모든 위치에서 도착지과 골대사이의 코사인값
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_end_coo_bin = np.concatenate((x0_ball_end, y0_ball_end))
        a = goal_coo_bin - coords
        b = ball_end_coo_bin - coords
        matrix[10, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 11: Sine of the angle between the ball and goal
        # 모든 위치에서 도착지과 골대사이의 사인값
        matrix[11, :, :] = np.sqrt(1 - matrix[10, :, :] ** 2)

        matrix[12, :, :] = ball_height_ground
        matrix[13, :, :] = ball_height_low
        matrix[14, :, :] = ball_height_high
  
        for i in range(len(matrix)):
            matrix[i, :, :] = scaler.fit_transform(matrix[i, :, :])
        # Mask
        mask = np.zeros((1, self.y_bins, self.x_bins))

        end_ball_coo = np.array([[intended_end_x, intended_end_y]])
        #end_ball_coo = np.array([[end_x, end_y]])
        
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1

        if target is not None:
            return (
                torch.from_numpy(matrix).float(),
                torch.from_numpy(mask).float(),
                torch.tensor([target]).float(),
            )
        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            None,
        )