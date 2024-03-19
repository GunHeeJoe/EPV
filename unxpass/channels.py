import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import math
from shapely.geometry import LineString
from unxpass.components.soccermap import SoccerMap
from sklearn.cluster import AgglomerativeClustering
from glob import glob

import re, os

class PassSuccessModel(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()
        self.model = SoccerMap(in_channels=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        return self.sigmoid(output)
    
def _is_inside_triangle(pnt, triangle):
    """Compute whether the given point is in the given triangle.

    Parameters
    ----------
    pnt : tuple (x, y)
        The given point.

    triangle : list of tuples [(x0, y0), (x1, y1), (x2, y2)]
        The corners of the triangle, clockwise.

    Returns
    -------
        Boolean
    """

    def _is_right_of(line):
        return (
            (line[1][0] - line[0][0]) * (pnt[1] - line[0][1])
            - (pnt[0] - line[0][0]) * (line[1][1] - line[0][1])
        ) <= 0

    return (
        _is_right_of([triangle[0], triangle[1]])
        & _is_right_of([triangle[1], triangle[2]])
        & _is_right_of([triangle[2], triangle[0]])
    )


def _get_passing_cone(start, end, dist=1):
    """Compute the corners of the triangular passing cone between the given start and end location.

    The cone starts at the start location and has a width of 2*dist at the end location, with the end location
    indicating the middle of the line that connects the two adjacent corners.

    Parameters
    ----------
    start : tuple (x, y)
        The given start location.

    end : tuple (x, y)
        The given end location.

    dist : int
        The distance between the end location and its two adjacent corners of the triangle.

    Returns
    -------
    List of tuples [(x0, y0), (x1, y1), (x2, y2)] containing the corners of the triangle, clockwise.

    """

    if (start[0] == end[0]) | (start[1] == end[1]):
        slope = 0
    else:
        #slope : 기울기 = dy/dx
        slope = (end[1] - start[1]) / (end[0] - start[0])

    dy = math.sqrt(dist**2 / (slope**2 + 1))
    dx = -slope * dy

    if start[0] == end[0]:  # have treated vertical line as horizontal one, rotate
        dx, dy = dy, dx

    pnt1 = (end[0] + dx, end[1] + dy)
    pnt2 = (end[0] - dx, end[1] - dy)
    return [start, pnt1, pnt2]


def nb_opp_in_path(opponents_coo, start_coo, end_coo, path_width= 3):
    """Getthe  number of opponents in the path between the start and end location of a pass.
    #면적까지 고려해서 지나친 선수의 수를 고려한 것 으로 패스 사이에 있다고해도
    triangular안에 있어야 선수가 카운트됨
    The path is defined as a triangular corridor between the pass origin and
    the receiver's location with a base of `x` meters at the receiver's
    location.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.
    path_width : float
        The width (in meters) of the triangular path at the receiver's location.

    Returns
    -------
    Features
        The number of opponents in the path of each pass.
    """

    start_coo = [start_coo[0], start_coo[1]]
    end_coo = [end_coo[0], end_coo[1]]

    #print("before : ",opponents_coo)
    opponents_coo = [(o[0]+0.5, o[1]+0.5) for o in opponents_coo]
    #print("after :",opponents_coo)

    triangle = _get_passing_cone(start_coo, end_coo, path_width)
    value = sum(_is_inside_triangle(o, triangle) for o in opponents_coo)

    return value


def _defenders_in_radius(sample, end_coo, coo, radius=3):
    """Get the number of defenders in a radius around the actions start and end location.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.
    radius : int
        The radius (in meters) of the circle in which defenders are counted.

    Returns
    -------
    Features
        The number of defenders is a radius around the actions start and end location.
    """
  
    try : 
        end_coo = np.array([end_coo[0], float(end_coo[1])])
        if coo == 'attacking':
            coord = np.array([(o["x"], o["y"]) for o in sample["freeze_frame_360_a0"] if o["teammate"]])
        elif coo == 'opponents':
            coord = np.array([(o["x"], o["y"]) for o in sample["freeze_frame_360_a0"] if not o["teammate"]])
        else:
            print("error")
            exit()

        if len(coord) == 0:
            return 0
        
        # Distance to every location
        dist_defender_start = np.sqrt(
            (coord[:, 0] - end_coo[0]) ** 2 + (coord[:, 1] - end_coo[1]) ** 2
        )
        #시작위치에서 radius안에 있는 수비수들의 수
        value = np.sum(dist_defender_start <= radius)
    except : 
        print("_defenders_in_radius error")
        print(f"coo = {coo}")
        print(f"end_coo={end_coo}")
        print(f"coord={coord}")
        print(f"sample ={sample}")
        print(f"dist_defender_start={dist_defender_start}")
        print(f"value={value}")
        

    return value


def dist_defender(sample, start_coo, end_coo, coo):
    """Get the distance to the nearest defender.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The distance to the nearest defender at the start and end location of the action,
        and along the action's path.
    """

    try : 
        start_coo = np.array([start_coo[0], start_coo[1]])
        end_coo = np.array([end_coo[0], float(end_coo[1])])

        if coo == 'attacking':
            coord = np.array([(o["x"], o["y"]) for o in sample["freeze_frame_360_a0"] if o["teammate"]])
        elif coo == 'opponents':
            coord = np.array([(o["x"], o["y"]) for o in sample["freeze_frame_360_a0"] if not o["teammate"]])
        else:
            print("error")
            exit()

        #수비수 or 공격수가 포착 안되는 경우는 거리를 측정할 수 없으므로
        #평균적으로 중앙(34,52)에 선수가 위치한다고 가정
        if len(coord) == 0:
            coord = np.array([(34,52)])
        
        # 패스 시작 - 상대팀간의 최소거리
        # Distance to start location
        end_nearest_distance = np.amin(np.sqrt((coord[:, 0] - end_coo[0]) ** 2 + (coord[:, 1] - end_coo[1]) ** 2))

        # Distance to action path
        # normalized tangent vector
        if (start_coo == end_coo).all():
            dist_defender_action = end_nearest_distance
        else:
            #시작->끝을 잇는 단위벡터
            d = np.divide(end_coo - start_coo, np.linalg.norm(end_coo - start_coo))
            
            # signed parallel distance components
            # s,t : 시작위치-선수를 잇는 벡터 - 단위벡터를 내적함으로써
            # 패스 경로상에 상대적으로 어느쪽에 위치하는지를 파악
            s = np.dot(start_coo - coord, d)
            t = np.dot(coord - end_coo, d)
            
            # clamped parallel distance
            # maximum.reduce : s VS t를 비교할 때, 둘 다 음수이면 0, 양수가 있으면 큰 값
            # 선수가 pass시작부분에 위치하면, s가 더 클거고, 반대면 t가 더 클거임
            h = np.maximum.reduce([s, t, np.zeros(len(coord))])
            
            # perpendicular distance component, as before
            # note that for the 3D case these will be vectors
            c = np.cross(coord - start_coo, d)
            # use hypot for Pythagoras to improve accuracy
            # np.hypot : h,c를 밑변 높이로 하는 직각삼각형의 빗변 구함
            dist_defender_action = np.amin(np.hypot(h, c))
    except:
        print("dist_defender error")
        print(f"coord={coord}")
        print(f"end_nearest_distance = {end_nearest_distance}")
        print(f"dist_defender_action = {dist_defender_action}")
    return end_nearest_distance, dist_defender_action


def angle_defender(sample, start_coo, end_coo):
    try :
        start_coo = np.array([start_coo[0], start_coo[1]])
        end_coo = np.array([end_coo[0], float(end_coo[1])])
        opponents_coo = np.array([(o["x"], o["y"]) for o in sample["freeze_frame_360_a0"] if not o["teammate"]])

        #수비수 or 공격수가 포착 안되는 경우는 거리를 측정할 수 없으므로
        #평균적으로 중앙(34,52)에 선수가 위치한다고 가정
        if len(opponents_coo) == 0:
            opponents_coo = np.array([(34,52)])

        pass_vector = np.array(end_coo) - np.array(start_coo)
        player_vectors = np.array(opponents_coo) - np.array(start_coo)

        angles = []
        for player_vector in player_vectors:
            dot_product = np.dot(pass_vector, player_vector)
            norm_product = np.linalg.norm(pass_vector) * np.linalg.norm(player_vector)

            cos_theta = dot_product / norm_product
            #연산과정에서 -1.0000002와 같이 -1 or 1를 약간 넘어가는 부분은
            #math.arccos 오류가 발생함으로 예외처리해줌
            if cos_theta <-1: cos_theta = -1
            if cos_theta > 1: cos_theta = 1
            
            angle = np.arccos(cos_theta)  
            angles.append(angle)
     
        min_angle = np.degrees(min(angles))
    except :
        print("angle_defender error")
        print(f"player_vectors_list = {player_vectors}")
        print(f"min_angle = {min_angle}")
    return min_angle

def get_pass_success_probability(matrix):
    #패스 성공 확률 모델에서는 13개의 채널을 사용한 SoccerMap Architecture를 사용하므로 이에 맞춰서 input수정
    # torch.from_numpy : 모델에 input으로 사용하기위해 tensor로 변환
    # unsqueeze(dim=0) : (batch, channel, 높이:72, 너비:108)에 맞춰서 변환
    input = torch.from_numpy(matrix[:13, :, :]).unsqueeze(dim=0)

    checkpoint_path = find_checkpoint()

    model = PassSuccessModel(in_channels=13)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with torch.no_grad():
        # model의 input으로 double타입 사용불가능
        output = model(input.float())

    # output : (1, 1, 72, 108)형태이므로 dim=0과1 차원은 제거
    return torch.squeeze(output, dim=[0,1])

def find_checkpoint():
    # 체크포인트 디렉토리 설정
    base_path = '../runs/pass_success'

    # 해당 경로에 있는 모든 'version' 폴더를 찾음
    version_dirs = glob(os.path.join(base_path, 'version*'))

    # 가장 최근 버전 폴더 찾기 (가장 높은 숫자를 가진 폴더)
    latest_version_dir = max(version_dirs, key=lambda x: int(x[-1]))

    # 해당 버전 폴더 내의 모든 '.ckpt' 파일을 찾음
    ckpt_files = glob(os.path.join(latest_version_dir, '*.ckpt'))

    return ckpt_files[0]

def get_pressure_line(xx, x_bin_def, y_bins, k=3):

    # CH16 : Dynamic Pressure line
    # 본 연구에서는 vertical pressure line를 사용하고, 각 셀의 값은 가장 가까운 압박라인의 인덱스를 활용함
    # clustering : 완전연결법 & k=3울 사용
    # team : 수비팀의 압박라인을 사용하고, 골키퍼는 제외
    cluster_model = AgglomerativeClustering(n_clusters=k, linkage='complete')
    opponent_bin_sorted = np.sort(x_bin_def+0.5).reshape(-1,1)[:-1]

    # 각 수비수들의 좌표에 대응되는 클러스터(0, 1, 2)를 추출
    labels = cluster_model.fit_predict(opponent_bin_sorted)
    
    # 각 클러스터에 수비수들의 좌표의 평균 -> 각 클러스터의 평균적인 x좌표
    # 각 압박라인(인덱스)에 해당하는 x좌표 추출 -> 1:공격라인 & 2:미드필더라인 & 3:수비라인
    cluster_center_x = np.array(sorted([np.mean(opponent_bin_sorted[labels == i]) for i in range(k)]))

    # 3가지 압박라인과 각 셀간의 거리
    distances = np.abs(xx - cluster_center_x[:, np.newaxis])

    # 가장 가까운 압박라인의 인덱스 : 1부터 시작
    nearest_pressure_line_indices = np.argmin(distances, axis=0) + 1

    # 세로 방향(y축)이 같은 셀은 동일한 pressure line index를 갖도록 설정함
    # ex) x=0.5인 셀은 y축이 무엇이든 같은 pressure_line를 가짐
    return np.tile(nearest_pressure_line_indices, (y_bins, 1))