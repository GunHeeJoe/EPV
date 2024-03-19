"""Data visualisation."""
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
from socceraction.spadl.config import field_length, field_width
import matplotlib.lines as mlines
from .preprocess_data import freeze_left_to_right

def plot_action(
    action: pd.Series,
    surface=None,
    show_action=True,
    ax=None,
    surface_kwargs={},
    log_bool = False,
    home_team_id = "Home",
    field_dimen=(106.,68.)
) -> None:
    """Plot a SPADL action with 360 freeze frame.

    Parameters
    ----------
    action : pandas.Series
        A row from the actions DataFrame.
    surface : np.arry, optional
        A surface to visualize on top of the pitch.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
    surface_kwargs : dict, optional
        Keyword arguments to pass to the surface plotting function.
    """

    #이벤트를 수행하는 팀을 왼쪽에 배치시킴
    action = play_left_to_right(action, home_team_id, field_dimen)

    #A팀을 RED, B팀을 BLUE로 설정함
    # if action['from'][0] == 'A':
    #     team_color = "#C8102E"
    #     opponent_color = "#6CABDD"
    # else:
    #     team_color = "#6CABDD"
    #     opponent_color = "#C8102E"     

    result = "Success" if action['accurate'] == 1 else "Fail"
    freeze_frame = pd.DataFrame.from_dict(action["freeze_frame"],orient='index')

    teammate_locs = freeze_frame[freeze_frame.teammate & ~freeze_frame.ball]
    opponent_locs = freeze_frame[~freeze_frame.teammate & ~freeze_frame.ball]
    ball_loc = freeze_frame[freeze_frame.ball]
    event_player_loc = freeze_frame[freeze_frame.teammate & freeze_frame.actor]

    # set up pitch
    p = Pitch(pitch_type="custom", pitch_length=field_dimen[0], pitch_width=field_dimen[1])

    if ax is None:
        _, ax = p.draw(figsize=(12, 8))
    else:
        p.draw(ax=ax)

    if show_action:
        p.arrows(
            ball_loc["start_x"],
            ball_loc["start_y"],
            ball_loc["end_x"],
            ball_loc["end_y"],
            color="black",
            headwidth=7,
            headlength=5,
            width=2,
            ax=ax,
        )
        

    # plot freeze frame
    p.scatter(teammate_locs.start_x, teammate_locs.start_y, c='#C8102E', s=80, ec="k", ax=ax)
    p.scatter(opponent_locs.start_x, opponent_locs.start_y, c='#6CABDD', s=80, ec="k", ax=ax)
    p.scatter(event_player_loc.start_x, event_player_loc.start_y, c="#FFFF00", s=80, ec="k", ax=ax)

    p.scatter(ball_loc.start_x, ball_loc.start_y, c="green", s=80, ec="k", ax=ax)
    p.scatter(ball_loc.end_x, ball_loc.end_y, c="green", s=80, ec="k", ax=ax)

    ax.quiver( teammate_locs.start_x, teammate_locs.start_y,teammate_locs.start_vx,teammate_locs.start_vy, color='#C8102E', scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=0.7)
    ax.quiver( opponent_locs.start_x, opponent_locs.start_y,opponent_locs.start_vx,opponent_locs.start_vy, color='#6CABDD', scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=0.7)
    
    for player in teammate_locs.index:
        ax.text(teammate_locs.at[player,'start_x']+0.5,teammate_locs.at[player,'start_y']+0.5,s=player,color='#C8102E')

    for player in opponent_locs.index:
        ax.text(opponent_locs.at[player,'start_x']+0.5,opponent_locs.at[player,'start_y']+0.5,s=player,color='#6CABDD')

    #plot surface
    if surface is not None:
        if log_bool:
            # img = ax.imshow(np.log(surface), extent=[0.0, field_dimen[0], 0.0, field_dimen[1]], origin="lower", cmap=plt.get_cmap('coolwarm'), **surface_kwargs)
            img = ax.imshow(np.log(surface), origin="lower", cmap=plt.get_cmap('coolwarm'), **surface_kwargs)
        else:
            # img = ax.imshow(surface, extent=[0.0, field_dimen[0], 0.0, field_dimen[1]], origin="lower", cmap=plt.get_cmap('coolwarm'),**surface_kwargs)
            img = ax.imshow(surface, origin="lower", cmap=plt.get_cmap('coolwarm'),**surface_kwargs)
        
        x_bin, y_bin = _get_cell_indexes(ball_loc.end_x,ball_loc.end_y, field_dimen=field_dimen)
        x_bin = x_bin.values[0]
        y_bin = y_bin.values[0]
        probability = surface[y_bin][x_bin]

        title = f"{result} {action['eventName']}({action['subEventName']})\n{action['from']} -> {action['to']}\nend-location = {probability:.5f}{x_bin,y_bin}"
        ax.set_title(f"{title}")
        plt.colorbar(img,ax=ax)
    else:
        ax.set_title(f"{result} {action['eventName']}({action['subEventName']})\n{action['from']} -> {action['to']}")

    red_dot = mlines.Line2D([], [], color='#C8102E', marker='o', linestyle='None', markersize=7, label='Attacking team')
    blue_dot = mlines.Line2D([], [], color='#6CABDD', marker='o', linestyle='None', markersize=7, label='Defending team')
    black_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=7, label='Pass')
    yellow_dot = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None', markersize=7, label='Event Player')

    # 범례에 핸들과 레이블을 지정합니다.
    ax.legend(handles=[red_dot, blue_dot, black_dot, yellow_dot], loc='upper right', fontsize='x-small')

    return ax

def _get_cell_indexes(x, y, field_dimen):
    x_bin = np.clip(x, 0, field_dimen[0] - 1).astype(np.uint8)
    y_bin = np.clip(y, 0, field_dimen[1] - 1).astype(np.uint8)

    return x_bin, y_bin

#이벤트(패스)를 수행하는 팀을 왼쪽에 배치시킴
def play_left_to_right(actions,home_team_id,field_dimen):
    ltr_actions = actions.copy()

    #away(B)팀이 이벤트를 수행하고 있으면, 오른쪽에 있던 원정팀을 flip시킴
    away_id = ltr_actions.team != home_team_id

    #이벤트데이터의 시작/끝위치 대칭
    if away_id:
        for col in ["start_x", "end_x"]:
            ltr_actions[col] = field_dimen[0] - ltr_actions[col]
        for col in ["start_y", "end_y"]:
            ltr_actions[col] = field_dimen[1] - ltr_actions[col]

        ltr_actions['freeze_frame'] = freeze_left_to_right(ltr_actions,field_dimen)
    return ltr_actions


