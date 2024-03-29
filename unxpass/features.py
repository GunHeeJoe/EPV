"""Implements the features used in each compoment."""
import math
from functools import partial, reduce, wraps
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import socceraction.vaep.features as fs
from rich.progress import track

from socceraction.spadl import config as spadl
from socceraction.spadl.utils import add_names
from socceraction.vaep.features import gamestates as to_gamestates

from unxpass.databases import Database
from unxpass.databases.base import TABLE_ACTIONS
from unxpass.utils import play_left_to_right


_spadl_cfg = {
    "length": 105,
    "width": 68,
    "penalty_box_length": 16.5,
    "penalty_box_width": 40.3,
    "six_yard_box_length": 5.5,
    "six_yard_box_width": 18.3,
    "goal_width": 7.32,
    "penalty_spot_distance": 11,
    "goal_length": 2,
    "origin_x": 0,
    "origin_y": 0,
    "circle_radius": 9.15,
}

_goal_x: float = _spadl_cfg["length"]
_goal_y: float = _spadl_cfg["width"] / 2

_pass_like = [
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
]


def required_fields(fields):
    def decorator(func):
        func.required_fields = fields
        return func

    return decorator

#intended_endlocation울 구한 dataframe에서 
#end_location을 기준으로 actionfn이라는 feature를 계산하는 원리로
#예를들어) intended(relative_end)
def intended(actionfn):
    """Make a function decorator to apply actionfeatures to intended end locations.

    This decorator replaces the end location of failed passes with the location
    of the most likely receiver before applying the feature transformer.

    The intended receiver is determined as the player that is closest to where
    the ball was intercepted and has the smallest angle to the line of the
    pass [1].

    Parameters
    ----------
    actionfn : callable
        A feature transformer that operates on gamestates.

    Returns
    -------
    FeatureTransfomer
        A feature transformer that operates on the intended end location.

    References
    ----------
    .. [1] Power, Paul, et al. "Not all passes are created equal: Objectively
       measuring the risk and reward of passes in soccer from tracking data."
       Proceedings of the 23rd ACM SIGKDD international conference on knowledge
       discovery and data mining. 2017.
    """
    # FIXME
    #아래에서 intend이름 수정해서 넣어놔서 여기서 한번 더 추가하면
    #intended_intended_로 바껴서 에러뜨니 주의
    #actionfn.__name__ = f"intended_{actionfn.__name__}"
    
    #𝐸𝑥𝑝𝑒𝑐𝑡𝑒𝑑 𝑟𝑒𝑐𝑒𝑖𝑣𝑒를 계산하기위한 작업
    #(Min-Distance / Distance) x (Min-Angle / Angle)
    # @wraps(actionfn)
    def _wrapper(gamestates) -> pd.DataFrame:
        if not isinstance(gamestates, (list,)):
            gamestates = [gamestates]
        actions = gamestates[0].copy()
        failed_passes = actions[
            actions["type_name"].isin(_pass_like) & (actions["result_name"] != "success")
        ]
        
        for idx, action in failed_passes.iterrows():
            if action["freeze_frame_360"] is None:
                continue
            # get coordinates of the pass start location, interception point
            # and each potential receiver
            receiver_coo = np.array(
                [
                    (o["x"], o["y"])
                    for o in action["freeze_frame_360"]
                    if o["teammate"] and not o["actor"]
                ]
            )
            if len(receiver_coo) == 0:
                continue
            ball_coo = np.array([action.start_x, action.start_y])
            interception_coo = np.array([action.end_x, action.end_y])
            # compute the distance between the location where the ball was
            # intercepted and each potential receiver
            dist = np.sqrt(
                (receiver_coo[:, 0] - interception_coo[0]) ** 2
                + (receiver_coo[:, 1] - interception_coo[1]) ** 2
            )
            # compute the angle between each potential receiver and the passing line
            a = interception_coo - ball_coo
            b = receiver_coo - ball_coo
            angle = np.arccos(
                np.clip(
                    np.sum(a * b, axis=1) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)), -1, 1
                )
            )
            # if not players are in 20 degrees of the pass line, the intended
            # receiver was probably not in the freeeze frame
            if np.amin(angle) > 0.35:
                continue
            # only consider players in 20 degrees of the pass line
            too_wide = np.where(angle > 0.35)[0]
            dist[too_wide] = np.inf
            # find the most likely intended receiver
            # TODO: you could play around with the weight given to the distance
            # and angle here
            exp_receiver = np.argmax((np.amin(dist) / dist) * (np.amin(angle) / angle))
            actions.loc[idx, ["end_x", "end_y"]] = receiver_coo[exp_receiver]
        
        #end_location -> intended_end_location으로 수정한 데이터프레임에서
        #원하는 actionfn을 반환하는 코드
        return actionfn([actions] + gamestates[1:])

    return _wrapper

def feature_column_names(fs: List[Callable], nb_prev_actions: int = 3) -> List[str]:
    """Return the names of the features generatbyed  a list of transformers.

    Parameters
    ----------
    fs : list(callable)
        A list of feature transformers.
    nb_prev_actions : int, default=3
        The number of previous actions included in the game state.

    Returns
    -------
    list(str)
        The name of each generated feature.
    """
    cols = TABLE_ACTIONS + ["type_name", "result_name", "bodypart_name"]
    dummy_actions = pd.DataFrame(np.zeros((10, len(cols))), columns=cols).set_index(
        ["game_id", "action_id"]
    )
    for c in cols:
        if "name" in c:
            dummy_actions[c] = dummy_actions[c].astype(str)
    gs = to_gamestates(dummy_actions, nb_prev_actions)  # type: ignore
    
    return list(pd.concat([f(gs) for f in fs], axis=1).columns.values)


actiontype = required_fields(["type_id"])(fs.actiontype)
actiontype_onehot = required_fields(["type_name"])(fs.actiontype_onehot)
result = required_fields(["result_id"])(fs.result)
result_onehot = required_fields(["result_name"])(fs.result_onehot)
actiontype_result_onehot = required_fields(["type_name", "result_name"])(
    fs.actiontype_result_onehot
)
bodypart = required_fields(["bodypart_id"])(fs.bodypart)
bodypart_onehot = required_fields(["bodypart_name"])(fs.bodypart_onehot)
time = required_fields(["period_id", "time_seconds"])(fs.time)
startlocation = required_fields(["start_x", "start_y"])(fs.startlocation)
endlocation = required_fields(["end_x", "end_y"])(fs.endlocation)
#intended_endlocation = required_fields(["intended_end_x", "intended_end_y"])(fs.endlocation)
endpolar = required_fields(["end_x", "end_y"])(fs.endpolar)
startpolar = required_fields(["start_x", "start_y"])(fs.startpolar)
movement = required_fields(["start_x", "start_y", "end_x", "end_y"])(fs.movement)
team = required_fields(["team_id"])(fs.team)
time_delta = required_fields(["time_delta"])(fs.time_delta)
space_delta = required_fields(["start_x", "start_y", "end_x", "end_y"])(fs.space_delta)
goalscore = required_fields(["team_id", "type_name", "result_id"])(fs.goalscore)

@required_fields(["end_x", "end_y"])
@fs.simple
def intended_endpolar(actions): 
    polardf = pd.DataFrame(index=actions.index)
    dx = (_goal_x - actions['end_x']).abs().values
    dy = (_goal_y - actions['end_y']).abs().values
    polardf['intended_end_dist_to_goal'] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        polardf['intended_end_angle_to_goal'] = np.nan_to_num(np.arctan(dy / dx))
    return polardf

@required_fields(["end_x", "end_y"])
@fs.simple
def intended_endlocation(actions): 
    location_df = actions[['end_x', 'end_y']].rename(columns={'end_x':'intended_end_x',
                                                              'end_y':'intended_end_y'
                                                              })
    return location_df

@required_fields(["start_x", "start_y", "end_x", "end_y"])
@fs.simple
def intended_movement(actions): 
    actions['intended_dx'] = actions.end_x - actions.start_x
    actions['intended_dy'] = actions.end_y - actions.start_y
    actions['intended_movement'] = np.sqrt(actions['intended_dx']**2 + actions['intended_dy']**2)
    
    return actions[['intended_dx','intended_dy','intended_movement']]

#시작좌표를 기준으로 sideline, goalline을 계산하는 작업 
@required_fields(["start_x", "start_y"])
@fs.simple
def relative_startlocation(actions):
    """Get the location where each action started relative to the sideline and goalline.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'start_dist_sideline' and 'start_dist_goalline' of each action.
    """
    
    actions["_start_dist_sideline"] = _spadl_cfg["width"] - actions["start_y"]
    actions["start_dist_sideline"] = actions[["_start_dist_sideline", "start_y"]].min(axis=1)
    actions["start_dist_goalline"] = _spadl_cfg["length"] - actions["start_x"]
    return actions[["start_dist_sideline", "start_dist_goalline"]]


#끝좌표를 기준으로 sideline, goalline을 계산하는 작업 
@required_fields(["end_x", "end_y"])
@fs.simple
def relative_endlocation(actions):
    """Get the location where each action ended relative to the sideline and goalline.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'end_dist_sideline' and 'end_dist_goalline' of each action.
    """
    actions["_end_dist_sideline"] = _spadl_cfg["width"] - actions["end_y"]
    actions["end_dist_sideline"] = actions[["_end_dist_sideline", "end_y"]].min(axis=1)
    actions["end_dist_goalline"] = _spadl_cfg["length"] - actions["end_x"]
    return actions[["end_dist_sideline", "end_dist_goalline"]]

#끝좌표를 기준으로 sideline, goalline을 계산하는 작업 
@required_fields(["end_x", "end_y"])
@fs.simple
def intended_relative_endlocation(actions):
    actions["_end_dist_sideline"] = _spadl_cfg["width"] - actions["end_y"]
    actions["intended_end_dist_sideline"] = actions[["_end_dist_sideline", "end_y"]].min(axis=1)
    actions["intended_end_dist_goalline"] = _spadl_cfg["length"] - actions["end_x"]

    return actions[["intended_end_dist_sideline", "intended_end_dist_goalline"]]


#볼과 골까지의 각도
@required_fields(["start_x", "start_y", "end_x", "end_y"])
@fs.simple
def angle(actions):
    """Get the angle between the start and end location of an action.

    The action's start location is used as the origin in a polar coordinate
    system with the polar axis parallell to the the goalline.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'angle' of each action.
    """
    
    polardf = pd.DataFrame(index=actions.index)
    
    dx = (actions["end_x"] - actions["start_x"]).values
    dy = (actions["end_y"] - actions["start_y"]).values

    #atan2 로 계산하면, 시계방향:pi & 반시계방향:-pi으로 angle이 설정됨
    polardf["angle"] = [math.atan2(float(dy[i]), float(dx[i])) for i in range(len(polardf))] 
    
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     #arctan : 부호정보를 사용하지 않고 하나의 인자로 처리
    #     #arctan2 : 부호정보를 사용하고, 두 개의 인자로 처리
    #     polardf["angle"] = np.nan_to_num(np.arctan(dy / dx))
    # polardf.loc[actions["end_x"] < actions["start_x"], "angle"] += np.pi
    return polardf



#intended(intended_angle) 실행시 intended_endlocation이 end_location으로 바뀌어있음
@required_fields(["start_x", "start_y", "end_x", "end_y"])
@fs.simple
def intended_angle(actions):
    polardf = pd.DataFrame(index=actions.index)
    
    dx = (actions["end_x"] - actions["start_x"]).values
    dy = (actions["end_y"] - actions["start_y"]).values
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["intended_angle"] = np.nan_to_num(np.arctan(dy / dx))
    polardf.loc[actions["end_x"] < actions["start_x"], "intended_angle"] += np.pi

    return polardf

#추정된 볼의 속도 추정이라고는 썻는데, 아닌거 같음
#패스하기 전 선수가 공을 몰고간 시간이라고 생각함
#이전액션의 end과 현재액션의 start를 뺀다는 거 자체가
#이전액션->패스전까지 액션으로 몰고갔을 때 거리를 말하는건데
#즉 다이렉트 패스 : a1의end=a0의start가 같을 때는, dx=dy=0이므로 speed=0(이게 공의 속도를 말하는게 아니잖아)
#동일선수가 드리블(a1) 후  패스(a0)라면 : a1'end=a0'start이므로 이또한 speed=0(이것도 공의 속도를 말하는게 아니잖아)
#  
# @required_fields(["start_x", "start_y", "end_x", "end_y", "time_delta"])
# def speed(gamestates):
#     """Get the speed at which the ball moved during the previous actions.

#     Parameters
#     ----------
#     gamestates : GameStates
#         The game states of a game.

#     Returns
#     -------
#     Features
#         A dataframe with columns 'speedx_a0i', 'speedy_a0i', 'speed_a0i'
#         for each <nb_prev_actions> containing the ball speed in m/s  between
#         action ai and action a0.
#     """
#     #해당 데이터의 첫번째 game_state
#     #speed_a01 : 두번째 - 첫번째 간의 거리/시간=속력
#     #speed_a02 : 세번째 - 첫번째 간의 속력
#     #speed_a03 : 네번째 - 첫번째 간의 속력
#     a0 = gamestates[0]
#     speed = pd.DataFrame(index=a0.index)

#     for i, a in enumerate(gamestates[1:]):
#         #dx, dy,dt구하는 과정 자체가 잘못된거 같은데
#         #dx,dy는 현재액션의 패스길이인데, 아래와 같이 구하면 과거끝-현재시작으로 계산되니까
#         #과거의 패스길이가 계산되어버리잖아
#         #그러나 현재의 데이터에서 과거의정보만 불어올 수 있고, 미래의 정보를 불러올 수가 없음
#         #그래서 과거의 정보를 넣고 마지막에 각 speed값을 하나씩 내리는 방식으로 전개해보자
#         dx = a.end_x - a0.start_x
#         dy = a.end_y - a0.start_y
        
#         #dt도 과거 패스의 시간으로 계산되어버리잖아
#         #우리는 현재패스의 거리/시간정보를 알아야함
#         dt = a0.time_seconds - a.time_seconds
#         dt[dt <= 0] = 1e-6
        
#         speed["speedx_a0" + (str(i + 1))] = dx.abs() / dt
#         speed["speedy_a0" + (str(i + 1))] = dy.abs() / dt
#         speed["speed_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2) / dt
        
#     #오류데이터이므로 반드시 전처리해줘야함
#     #speed같은 경우 이전 액션과의 거리/시간=속력으로 계산되므로
#     #이전 액션이 다른 경기, 하프타임등의 이유로 거리 or 시간이 바뀌면 
#     #이상한 데이터로 바뀌므로 따로 전처리해줘야함
#     #실제로 패스의 속도는 30<speed<1000 or 음수값은 없는데
#     #10000이상의 패스 속도가 존재하므로 전처리해줘야함
#     return speed

@required_fields(["start_x", "start_y", "end_x", "end_y", "time_seconds"])
#a1의 패스속도, a2의 패스속도
def speed(gamestates):
    a0 = gamestates[0]

    speed = pd.DataFrame(index=a0.index)

    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a.start_x
        dy = a.end_y - a.start_y
        
        dt = a0.time_seconds - a.time_seconds
        dt[dt <= 0] = 1e-6
        
        speed["speedx_a0" + (str(i + 1))] = dx.abs() / dt
        speed["speedy_a0" + (str(i + 1))] = dy.abs() / dt
        speed["speed_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2) / dt
        
    return speed

@required_fields(["freeze_frame_360"])
@fs.simple
def freeze_frame_360(actions):
    """Get the raw StatsBomb 360 freeze frame.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'freeze_frame_360' of each action.
    """
    if "freeze_frame_360" not in actions.columns:
        df = pd.DataFrame(index=actions.index)
        df["freeze_frame_360"] = None
        return df
    return actions[["freeze_frame_360"]]


@required_fields(["under_pressure"])
@fs.simple
def under_pressure(actions):
    """Get the value of StatsBomb's 'under_pressure' attribute.

    Every on-the-ball event that overlaps the duration of a pressure event is
    annotated as being 'under_pressure'. For example, if a pressure
    event appears before a pass, and the pressure’s timestamp plus its
    duration encompasses the pass’s timestamp, that pass is said to have been
    made under pressure. If a pressure event occurs after a pass, but before
    the end of the pass (as calculated by using its duration), that pass is
    said to have been received under pressure. Events which are naturally
    performed under pressure like duels, dribbles etc, all pick up the
    attribute, even in the absence of an actual pressure event. Carries can be
    pressured not just by pressure events, but other defensive events (defined
    in change 2.) that happen during or at the end of the carry

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'under_pressure' attribute of each action.
    """
    return actions[["under_pressure"]].fillna(False)


@required_fields(["period_id", "time_seconds", "player_id", "type_name"])
@fs.simple
def player_possession_time(actions):
    """Get the time (sec) a player was in ball possession before attempting the action.

    We only look at the dribble preceding the action and reset thepossession
    tim e after a defensive interception attempt or a take-on.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The 'player_possession_time' of each action.
    """
    #현재 액션 정보
    cur_action = actions[["period_id", "time_seconds", "player_id", "type_name"]]
    #이전 액션 정보
    prev_action = actions.shift(1)[["period_id", "time_seconds", "player_id", "type_name"]]
    df = cur_action.join(prev_action, rsuffix="_prev")
    #같은 선수, 같은 타임때 패스하기전 드리블의 소유 시간을 측정
    same_player = df.player_id == df.player_id_prev
    same_period = df.period_id == df.period_id_prev
    prev_dribble = df.type_name_prev == "dribble"
    mask = same_period & same_player & prev_dribble
    df.loc[mask, "player_possession_time"] = (
        df.loc[mask, "time_seconds"] - df.loc[mask, "time_seconds_prev"]
    )
    return df[["player_possession_time"]].fillna(0)


@required_fields(["extra"])
@fs.simple
def ball_height(actions):
    """Get the height of a pass.

    The height is defined as:
        - "ground": ball doesn’t come off the ground.
        - "low": ball comes off the ground but is under shoulder level at peak height.
        - "high": ball goes above shoulder level at peak height.

    This feature is only defined for pass-like actions.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The ball height during each pass-like action.
    """
    df = pd.DataFrame(index=actions.index)
    df["ball_height"] = None
    passes = actions[actions.type_name.isin(_pass_like)]
    for idx, pas in passes.iterrows():
        pass_height = pas["extra"]["pass"].get("height", {"name": None}).get("name")
        if pass_height == "Ground Pass":
            df.at[idx, "ball_height"] = "ground"
        elif pass_height == "Low Pass":
            df.at[idx, "ball_height"] = "low"
        elif pass_height == "High Pass":
            df.at[idx, "ball_height"] = "high"
        else:
            df.at[idx, "ball_height"] = "ground"
    return df


@required_fields(["extra"])
@fs.simple
def ball_height_onehot(actions):
    """Get the one_hot_encoded height of a pass.

    The height is defined as:
        - "ground": ball doesn’t come off the ground.
        - "low": ball comes off the ground but is under shoulder level at peak height.
        - "high": ball goes above shoulder level at peak height.

    This feature is only defined for pass-like actions.

    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The ball height during each pass-like action.
    """
    df = pd.DataFrame(index=actions.index)
    df["ball_height_ground"] = False
    df["ball_height_low"] = False
    df["ball_height_high"] = False
    passes = actions[actions.type_name.isin(_pass_like)]
    for idx, pas in passes.iterrows():
        pass_height = pas["extra"]["pass"].get("height", {"name": None}).get("name")
        if pass_height == "Ground Pass":
            df.at[idx, "ball_height_ground"] = True
        elif pass_height == "Low Pass":
            df.at[idx, "ball_height_low"] = True
        elif pass_height == "High Pass":
            df.at[idx, "ball_height_high"] = True
    return df


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


def  _get_passing_cone(start, end, dist=1):
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


@required_fields(["freeze_frame_360", "start_x", "start_y", "end_x", "end_y"])
@fs.simple
def nb_opp_in_path(actions, path_width: int = 1):
    """Get the number of opponents in the path between the start and end location of a pass.
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
    df = pd.DataFrame(index=actions.index)
    df["nb_opp_in_path"] = 0

    passes = actions[actions.type_name.isin(_pass_like)]
    for idx, pas in passes.iterrows():
        if not pas["freeze_frame_360"]:
            continue

        start_coo = [pas["start_x"], pas["start_y"]]
        end_coo = [pas["end_x"], pas["end_y"]]

        if start_coo == end_coo:
            continue

        opponents_coo = [(o["x"], o["y"]) for o in pas["freeze_frame_360"] if not o["teammate"]]
        triangle = _get_passing_cone(start_coo, end_coo, path_width)
        df.at[idx, "nb_opp_in_path"] = sum(_is_inside_triangle(o, triangle) for o in opponents_coo)
    return df


@required_fields(["freeze_frame_360", "start_x", "start_y", "end_x", "end_y"])
@fs.simple
def packing_rate(actions):
    """Get the number of defenders that are outplayed by a pass.
    #면적을 고려하지 않고 단순히, 상대 진영으로 가능 패스전-후사이에 선수의 유무만 파악한 방식
    #조건이 단순함으로 사람의 수는 nb_opp_in_path보다 좀 더 카운트 됨
    Parameters
    ----------
    actions : SPADLActions
        The actions of a game.

    Returns
    -------
    Features
        The number of defenders outplayed by a pass.
    """
    df = pd.DataFrame(index=actions.index)
    df["packing_rate"] = 0

    goal_coo = np.array([_goal_x, _goal_y])

    passes = actions[actions.type_name.isin(_pass_like)]
    for idx, pas in passes.iterrows():
        if not pas["freeze_frame_360"]:
            continue

        opponents_coo = np.array(
            [(o["x"], o["y"]) for o in pas["freeze_frame_360"] if not o["teammate"]]
        )
        if len(opponents_coo) == 0:
            continue

        ball_coo = np.array([pas["start_x"], pas["start_y"]])
        end_coo = np.array([pas["end_x"], pas["end_y"]])

        dist_ball_goal = np.sqrt(
            (goal_coo[0] - ball_coo[0]) ** 2 + (goal_coo[1] - ball_coo[1]) ** 2
        )
        dist_destination_goal = np.sqrt(
            (goal_coo[0] - end_coo[0]) ** 2 + (goal_coo[1] - end_coo[1]) ** 2
        )
        dist_def_goal = np.sqrt(
            (opponents_coo[:, 0] - goal_coo[0]) ** 2 + (opponents_coo[:, 1] - goal_coo[1]) ** 2
        )
        #해당 패스가 상대 수비를 몇명제쳤는지 세는 기준
        #패스시작-끝 사이에 선수가 존재하며, 패스는 상대 골쪽으로 향했을 때
        outplayed = (
            # The defender is between the ball and the goal before the pass
            (dist_def_goal <= dist_ball_goal)
            # The defender is further from the goal than the ball after the pass
            & (dist_def_goal > dist_destination_goal)
            # The ball moved closer to the goal
            & (dist_destination_goal <= dist_ball_goal)
        )
        df.at[idx, "packing_rate"] = np.sum(outplayed)
    return df


def _defenders_in_radius(actions, radius: int = 1):
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
    defenders_in_radius = np.zeros((len(actions), 2), dtype=int)
    for i, (_, action) in enumerate(actions.iterrows()):
        if not action["freeze_frame_360"]:
            continue

        opponents_coo = np.array(
            [(o["x"], o["y"]) for o in action["freeze_frame_360"] if not o["teammate"]]
        )
        if len(opponents_coo) == 0:
            continue

        start_coo = np.array([action["start_x"], action["start_y"]])
        end_coo = np.array([action["end_x"], action["end_y"]])

        # Distance to start location
        dist_defender_start = np.sqrt(
            (opponents_coo[:, 0] - start_coo[0]) ** 2 + (opponents_coo[:, 1] - start_coo[1]) ** 2
        )
        #시작위치에서 radius안에 있는 수비수들의 수
        defenders_in_radius[i, 0] = np.sum(dist_defender_start <= radius)

        # Distance to end location
        dist_defender_end = np.sqrt(
            (opponents_coo[:, 0] - end_coo[0]) ** 2 + (opponents_coo[:, 1] - end_coo[1]) ** 2
        )
        #끝위치에서 radius안에 있는 수비수들의 수
        defenders_in_radius[i, 1] = np.sum(dist_defender_end <= radius)

    return pd.DataFrame(
        defenders_in_radius,
        index=actions.index,
        columns=[f"nb_defenders_start_{radius}m", f"nb_defenders_end_{radius}m"],
    )


@required_fields(["freeze_frame_360", "start_x", "start_y", "end_x", "end_y"])
@fs.simple
def dist_defender(actions):
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
    dist = np.ones((len(actions), 3), dtype=float) * 20
    for i, (_, action) in enumerate(actions.iterrows()):
        if not action["freeze_frame_360"]:
            continue

        opponents_coo = np.array(
            [(o["x"], o["y"]) for o in action["freeze_frame_360"] if not o["teammate"]]
        )
        if len(opponents_coo) == 0:
            continue

        start_coo = np.array([action["start_x"], action["start_y"]])
        end_coo = np.array([action["end_x"], action["end_y"]])

        # 패스 시작 - 상대팀간의 최소거리
        # Distance to start location
        dist[i, 0] = np.amin(
            np.sqrt(
                (opponents_coo[:, 0] - start_coo[0]) ** 2
                + (opponents_coo[:, 1] - start_coo[1]) ** 2
            )
        )

        # 패스 목적지 - 상대팀간의 최소 거리
        # Distance to end location
        dist[i, 1] = np.amin(
            np.sqrt(
                (opponents_coo[:, 0] - end_coo[0]) ** 2 + (opponents_coo[:, 1] - end_coo[1]) ** 2
            )
        )

        # Distance to action path
        # normalized tangent vector
        if (start_coo == end_coo).all():
            dist[i, 2] = dist[i, 0]
        else:
            #시작->끝을 잇는 단위벡터
            d = np.divide(end_coo - start_coo, np.linalg.norm(end_coo - start_coo))
            
            # signed parallel distance components
            # s,t : 시작위치-선수를 잇는 벡터 - 단위벡터를 내적함으로써
            # 패스 경로상에 상대적으로 어느쪽에 위치하는지를 파악
            s = np.dot(start_coo - opponents_coo, d)
            t = np.dot(opponents_coo - end_coo, d)
            
            # clamped parallel distance
            h = np.maximum.reduce([s, t, np.zeros(len(opponents_coo))])
            # perpendicular distance component, as before
            # note that for the 3D case these will be vectors
            c = np.cross(opponents_coo - start_coo, d)
            # use hypot for Pythagoras to improve accuracy
            dist[i, 2] = np.amin(np.hypot(h, c))

    return pd.DataFrame(
        dist,
        index=actions.index,
        columns=["dist_defender_start", "dist_defender_end", "dist_defender_action"],
    )


#pass_options : 우리팀원에 대한 각각의 정보를 받기 위한 작업
# 즉, 보이는 팀원이 3명이면, dist_ball_to_teammate1,2,3 3개를 구하는 과정이다
# Sklearn, XGBoost는 pass_selection구할때 주어진 팀원에 대한 정보만 갖고있고 이를 classifier함으로
# 이러한 정보가 필요하지만, soccermap은 모든 pixel의 정보가 필요하므로 이렇게 특정 pixel만 구할 필요가없음
# 즉, soccermap에선 필요없음(이미 구현한 feature가 존재함)
#freeze_frame_360에서 얻을 수 있는 다양한 feature들
@required_fields(["freeze_frame_360", "start_x", "start_y"])
def pass_options(gamestates):
    """Get features for each passing option in the 360 freeze fram.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with the origin and destination location, pass distance,
        pass angle, angle to goal at origin and destination, distance to the
        nearest defender at the destination, and distance to the nearest
        defender to a straight line between origin and destination for each
        pass option in the 360 freeze frame.
    """
    a0 = gamestates[0]
    options = []
    passes = a0[a0.type_name.isin(_pass_like)]
    for (game_idx, action_idx), action in passes.iterrows():
        if action["freeze_frame_360"] is None:
            continue

        pass_options = [t for t in action["freeze_frame_360"] if t["teammate"] and not t["actor"]]
        opponents_coo = np.array(
            [(o["x"], o["y"]) for o in action["freeze_frame_360"] if not o["teammate"]]
        )
        ball_x, ball_y = action["start_x"], action["start_y"]
        origin_dx = abs(_goal_x - ball_x)
        origin_dy = abs(_goal_y - ball_y)
        for i, teammate in enumerate(pass_options):
            # Distance to goal
            destination_dx = abs(_goal_x - teammate["x"])
            destination_dy = abs(_goal_y - teammate["y"])
            # Pass distance
            dx = teammate["x"] - ball_x
            dy = teammate["y"] - ball_y
            # Pass angle
            angle = math.atan2(dy, dx)
            if teammate["x"] < ball_x:
                angle += math.pi
            if len(opponents_coo) > 0:
                # Distance to defender
                dists_defender = np.sqrt(
                    (opponents_coo[:, 0] - teammate["x"]) ** 2
                    + (opponents_coo[:, 1] - teammate["y"]) ** 2
                )
                # Distance to action path
                start_coo = np.array([action["start_x"], action["start_y"]])
                end_coo = np.array([teammate["x"], teammate["y"]], dtype=np.float64)
                # normalized tangent vector
                d = np.divide(end_coo - start_coo, np.linalg.norm(end_coo - start_coo))
                # signed parallel distance components
                s = np.dot(start_coo - opponents_coo, d)
                t = np.dot(opponents_coo - end_coo, d)
                # clamped parallel distance
                h = np.maximum.reduce([s, t, np.zeros(len(opponents_coo))])
                # perpendicular distance component, as before
                # note that for the 3D case these will be vectors
                c = np.cross(opponents_coo - start_coo, d)
                # use hypot for Pythagoras to improve accuracy
                dist_defender_action = np.hypot(h, c)
            else:
                dists_defender = [20]
                dist_defender_action = [20]
            options.append(
                {
                    "game_id": game_idx,
                    "action_id": action_idx,
                    "pass_option_id": i,
                    #공의 위치가 곧 액션의 시작위치라 가정
                    "origin_x": ball_x,
                    "origin_y": ball_y,
                    "destination_x": teammate["x"],
                    "destination_y": teammate["y"],
                    #공과 팀원 사이의 거리
                    "distance": math.sqrt(dx**2 + dy**2),
                    "angle": angle,
                    #시작or끝위치과 공 사이의 각도
                    "origin_angle_to_goal": math.atan2(origin_dy, origin_dx),
                    "destination_angle_to_goal": math.atan2(destination_dy, destination_dx),
                    #현재팀원과 상대팀 사이의 최소거리
                    "destination_distance_defender": np.amin(dists_defender),
                    #패스경로상의 상대팀 선수와 최소거리
                    "pass_distance_defender": np.amin(dist_defender_action),
                }
            )
    return pd.DataFrame(
        options,
        columns=[
            "game_id",
            "action_id",
            "pass_option_id",
            "origin_x",
            "origin_y",
            "destination_x",
            "destination_y",
            "distance",
            "angle",
            "origin_angle_to_goal",
            "destination_angle_to_goal",
            "destination_distance_defender",
            "pass_distance_defender",
        ],
    ).set_index(["game_id", "action_id"])


defenders_in_3m_radius = required_fields(
    ["start_x", "start_y", "end_x", "end_y", "freeze_frame_360"]
)(fs.simple(partial(_defenders_in_radius, radius=3)))
defenders_in_3m_radius.__name__ = "defenders_in_3m_radius"
defenders_in_5m_radius = required_fields(
    ["start_x", "start_y", "end_x", "end_y", "freeze_frame_360"]
)(fs.simple(partial(_defenders_in_radius, radius=5)))
defenders_in_5m_radius.__name__ = "defenders_in_5m_radius"

# Quick fix for FIXME above
# intended(함수) : 주어진 함수를 사용할 때는, intended_endlocation의 데이터를 활용함
# 즉, 주어진함수에서는 intended_end_location를 끝 위치로 가정하고 계산함
#동시에 두개의 데이터를 저장할 수 없음
#name을 변경하는 이유는 data name를 변경하기 위한것일 뿐 실제로 
intended_endlocation_function = intended(intended_endlocation)
intended_endlocation_function.__name__ = 'intended_endlocation_function'

intended_endpolar_function = intended(intended_endpolar)
intended_endpolar_function.__name__ = 'intended_endpolar_function'

intended_relative_endlocation_function = intended(intended_relative_endlocation)
intended_relative_endlocation_function.__name__ = 'intended_relative_endlocation_function'

intended_movement_function = intended(intended_movement)
intended_movement_function.__name__ = 'intended_movement_function'

intended_angle_function = intended(intended_angle)
intended_angle_function.__name__ = 'intended_angle_function'

#all_features : function을 전달함
all_features = [
    intended_endlocation_function,
    intended_endpolar_function,
    intended_relative_endlocation_function,
    intended_movement_function,
    intended_angle_function,
    actiontype,
    actiontype_onehot,
    result,
    result_onehot,
    actiontype_result_onehot,
    bodypart,
    bodypart_onehot,
    time,
    startlocation,
    relative_startlocation,
    endlocation,
    relative_endlocation,
    startpolar,
    endpolar,
    movement,
    team,
    time_delta,
    space_delta,
    goalscore,
    angle,
    under_pressure,
    packing_rate,
    ball_height,
    ball_height_onehot,
    player_possession_time,
    speed,
    nb_opp_in_path,
    dist_defender,
    freeze_frame_360,
    defenders_in_3m_radius,
    defenders_in_5m_radius
]

#주어진 pass action만 filtering하고
#모든 action을 left_to_right로
#ball-speed를 측정하기 위해서는 시계열데이터에서 미래의 정보가 필요함
def get_features(
    db: Database,
    game_id: int,
    xfns: List[Callable] = all_features,
    actionfilter: Optional[Callable] = None,
    nb_prev_actions: int = 3,
    overrides: Optional[pd.DataFrame] = None
):
    """Apply a list of feature generators.

    Parameters
    ----------
    db : Database
        The database with raw data.
    game_id : int
        The ID of the game for which features should be computed.
    xfns : List[Callable], optional
        The feature generators.
    actionfilter : Callable, optional
        A function that filters the actions to be used.
    nb_prev_actions : int, optional
        The number of previous actions to be included in a game state.
    overrides : pd.DataFrame, optional
        A dataframe with action attributes that override the values in the
        database. The dataframe should be indexed by game_id and action_id.

    Returns
    -------
    pd.DataFrame
        A dataframe with the features.
    """
    
    # retrieve actions from database
    actions = add_names(db.actions(game_id))

    if overrides is not None:
        actions.update(overrides)
    # filter actions of interest
    # 패스데이터의 game-id별 action-id의 인덱스를 추출함
    if actionfilter is None:
        idx = pd.Series([True] * len(actions), index=actions.index)
    else:
        idx = actionfilter(actions)


    # check if we have to return an empty dataframe
    # dataframe or xfns가 비어있을때 임시 대처법
    if idx.sum() < 1:
        column_names = feature_column_names(xfns, nb_prev_actions)
        return pd.DataFrame(columns=column_names)
    if len(xfns) < 1:
        return pd.DataFrame(index=actions.index.values[idx])
    
    # convert actions to gamestates
    home_team_id, _ = db.get_home_away_team_id(game_id)
    
    gamestates = play_left_to_right(to_gamestates(actions, nb_prev_actions), home_team_id)

    # compute features
    # 원래는 feature추출&인덱싱을 동시에 하는데
    # 미래의 데이터를 가져오려면 인덱싱을 먼저하면 안돼서 일단 생략
    # df_features = reduce(
    #     lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
    #     (fn(gamestates).loc[idx] for fn in xfns),
    # )

    df_features = reduce(
        lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
        (fn(gamestates) for fn in xfns),
    )

    #미래의 time_seconds를 현재 데이터에 저장하는 코드
    #마지막 액션의 경우는 다음 액션이 없으므로 현재데이터=미래데이터를 fill함
    if "time_seconds_a0" in df_features.columns:
        df_features['future_time_seconds'] = df_features['time_seconds_a0'].shift(-1)
        df_features['future_time_seconds'].fillna(df_features['time_seconds_a0'], inplace=True)

    #미래 데이터 가져왔으니 이제는 위에서 생략한 인덱싱 작업수행
    df_features = df_features[idx]

    # if we generated features for pass options instead of pass actions,
    # we need to add the pass option's ID to the index
    if "pass_option_id" in df_features.columns:
        df_features["pass_option_id"] = df_features["pass_option_id"].fillna(0).astype(int)
        df_features = df_features.set_index("pass_option_id", append=True)

    return df_features


def simulate_features(
    db: Database,
    game_id: int,
    xfns: List[Callable] = all_features,
    actionfilter: Optional[Callable] = None,
    nb_prev_actions: int = 3,
    xy: Optional[List[pd.DataFrame]] = None,
    x_bins: int = 104,
    y_bins: int = 68,
    result: Optional[str] = None,
):
    """Apply a list of feature generators.

    Parameters
    ----------
    db : Database
        The database with raw data.
    game_id : int
        The ID of the game for which features should be computed.
    xfns : List[Callable], optional
        The feature generators.
    actionfilter : Callable, optional
        A function that filters the actions to be used.
    nb_prev_actions : int, optional
        The number of previous actions to be included in a game state.
    xy: list(pd.DataFrame), optional
        The x and y coordinates of simulated end location.
    x_bins : int, optional
        The number of bins to simulated for the end location along the x-axis.
    y_bins : int, optional
        The number of bins to simulated for the end location along the y-axis.
    result : Optional[str], optional
        Sets the action result to be used for the simulation. If None, the
        actual result of the actions is used.

    Returns
    -------
    pd.DataFrame
        A dataframe with the features.
    """
    # retrieve actions from database
    actions = add_names(db.actions(game_id))
    # filter actions of interest
    if actionfilter is None:
        idx = pd.Series([True] * len(actions), index=actions.index)
    else:
        idx = actionfilter(actions)
    # check if we have to return an empty dataframe
    if idx.sum() < 1:
        column_names = feature_column_names(xfns, nb_prev_actions)
        return pd.DataFrame(columns=column_names)
    if len(xfns) < 1:
        return pd.DataFrame(index=actions.index.values[idx])
    # convert actions to gamestates
    home_team_id, _ = db.get_home_away_team_id(game_id)
    gamestates = play_left_to_right(to_gamestates(actions, nb_prev_actions), home_team_id)
    # simulate end location
    if xy is None:
        # - create bin centers
        # 각 pixel의 위치
        yy, xx = np.ogrid[0.5:y_bins, 0.5:x_bins]
        
        # - map to spadl coordinates
        # spadl로 좌표변환
        x_coo = np.clip(xx / x_bins * _spadl_cfg["length"], 0, _spadl_cfg["length"])
        y_coo = np.clip(yy / y_bins * _spadl_cfg["width"], 0, _spadl_cfg["width"])
        
    # simulate action result
    if result is not None:
        if result not in spadl.results:
            raise ValueError(f"Invalid result: {result}. Valid results are: {spadl.results}")
        gamestates[0].loc[:, ["result_id", "result_name"]] = (spadl.results.index(result), result)
    # compute fixed features
    xfns_fixed = [
        fn
        for fn in xfns
        if "end_x" not in fn.required_fields and "end_y" not in fn.required_fields
    ]
    df_fixed_features = reduce(
        lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
        (fn(gamestates).loc[idx] for fn in xfns_fixed),
    )
    # simulate other features
    xfns_to_simulates = [
        fn for fn in xfns if "end_x" in fn.required_fields or "end_y" in fn.required_fields
    ]
    df_simulated_features = []

    if xy is None:
        for end_x, end_y in track(
            np.array(np.meshgrid(x_coo, y_coo)).T.reshape(-1, 2),
            description=f"Simulating features for game {game_id}",
        ):
            gamestates[0].loc[:, ["end_x", "end_y"]] = (end_x, end_y)
            df_simulated_features.append(
                reduce(
                    lambda left, right: pd.merge(
                        left, right, how="outer", left_index=True, right_index=True
                    ),
                    (fn(gamestates).loc[idx] for fn in xfns_to_simulates),
                )
            )
    else:
        for end in xy:
            gamestates[0].loc[end.index, ["end_x", "end_y"]] = end.values
            df_simulated_features.append(
                reduce(
                    lambda left, right: pd.merge(
                        left, right, how="outer", left_index=True, right_index=True
                    ),
                    (fn(gamestates).loc[idx] for fn in xfns_to_simulates),
                )
            )

    df_features = pd.concat(df_simulated_features, axis=0).join(df_fixed_features, how="left")
    # if we generated features for pass options instead of pass actions,
    # we need to add the pass option's ID to the index
    if "pass_option_id" in df_features.columns:
        df_features["pass_option_id"] = df_features["pass_option_id"].fillna(0).astype(int)
        df_features = df_features.set_index("pass_option_id", append=True)
    return df_features
