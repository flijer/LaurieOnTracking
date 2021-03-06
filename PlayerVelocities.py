"""
    Calculate players speed in sprints
    Show charts

    @author: Jernej Flisar (@jernejfl)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def calc_players_velocity_in_sprints(tracking_home,teamname):
    sprint_threshold = 7
    sprint_window = 1 * 25

    players = np.unique([c.split('_')[1] for c in tracking_home.columns if c[:4] == teamname])
    players_df = pd.DataFrame(index=players)
    top_speeds = list()
    top_speeds_frame = list()
    players_sprints_start = list()
    players_sprints_end = list()
    for player in players_df.index:
        column = teamname+'_' + player + '_speed'
        top_speeds.append(tracking_home[
                              column].max())  # .sum()/25./1000 # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.
        top_speeds_frame.append(tracking_home[column].idxmax())
        # spring frames
        player_sprints = np.diff(1 * (np.convolve(1 * (tracking_home[column] >= sprint_threshold),
                                                  np.ones(sprint_window), mode='same') >= sprint_window)
                                 )
        players_sprints_start.append(np.where(player_sprints == 1)[0] - int(
            sprint_window / 2) + 1)  # adding sprint_window/2 because of the way that the convolution is centred
        players_sprints_end.append(np.where(player_sprints == -1)[0] + int(sprint_window / 2) + 1)

    players_df['top_speed[ms]'] = top_speeds
    players_df['top_speed[kmh]'] = [x * (60 * 60) / 1000 for x in players_df['top_speed[ms]']]
    players_df['top_speed_frame'] = top_speeds_frame
    players_df['sprints_start_frame'] = players_sprints_start
    players_df['sprints_end_frame'] = players_sprints_end

    return players_df


def speed_sprint_chart(playersdf, trackingdf, ax, fig, split=None, prefix='Home'):

    segment_ = list()
    dydxs = list()
    column = prefix + '_{}_speed_kmh'  # spped _speed_kmh
    column_x = prefix + '_{}_x'  # x position
    column_y = prefix + '_{}_y'  # y position

    for idx, row in playersdf[['sprints_start_frame', 'sprints_end_frame']].iterrows():

        for s, e in zip(row['sprints_start_frame'], row['sprints_end_frame']):

            x = trackingdf[column_x.format(idx)].iloc[s:e + 1].values  # Location x
            y = trackingdf[column_y.format(idx)].iloc[s:e + 1].values  # Location y

            #Naive split when players run on defense/offense
            if split == 'offense' and x[0] > x[-1]:
                continue
            if split == 'defense' and x[0] < x[-1]:
                continue

            # Player number - begining od plot
            ax.annotate(idx, xy=(trackingdf[column_x.format(idx)].iloc[s], trackingdf[column_y.format(idx)].iloc[s]),
                        color='white',
                        fontsize=14,
                        fontweight='bold',
                        ha="center"
                        )

            dydx = trackingdf[column.format(idx)].iloc[s:e + 1].values  # Speed

            points = np.array([x, y]).T.reshape(-1, 1, 2)

            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            for s in segments:
                segment_.append(s)
            for dx in dydx:
                if not np.isnan(dx): #prevent nan values
                    dydxs.append(dx)

    lc = LineCollection([x for x in segment_],
                        cmap=plt.get_cmap('OrRd'),  #colors
                        )

    lc.set_array(np.array(dydxs))
    lc.set_linewidth(5)

    line = ax.add_collection(lc)

    axcb = fig.colorbar(line, orientation="vertical", fraction=0.05, anchor=(0.0, 1.0))

    axcb.set_label('sprinting speed [km/h]')

    ax.set_title('Team {}\nPlayer Sprints {}'.format(prefix, split), fontsize=16)