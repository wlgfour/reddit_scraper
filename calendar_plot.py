"""
"""
import os
import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from helpers import read_data


# Helpers =====================================================
colors = [None, '#69B859', '#E76A47']  # [Irrelevant, +1, -1]
days = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
delta = datetime.timedelta(days=1)
def get_colors(Y):
    red = np.array(mpl.colors.to_rgba(colors[-1]))
    gre = np.array(mpl.colors.to_rgba(colors[1]))
    cols = np.zeros([Y.shape[0], 4])
    mask = np.tile((Y > 0)[:, np.newaxis], [1, 4])
    cols[mask] = gre
    cols[Y < 0] = red
    cols[:, -1] = 1 - 66/np.max(np.sqrt(Y), np.ones_like(Y)*75)
    return cols

def time_to_angle(time):
    if isinstance(time, datetime.datetime):
        seconds = (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    else:
        seconds = time.total_seconds()
    return 2 * np.pi * seconds / datetime.timedelta(days=1).total_seconds()

hours = [time_to_angle(datetime.timedelta(hours=i)) for i in range(0, 24, 3)]

# by subreddit =====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot a calendar plot where each day is a comment ' + \
                    'and each comment is a line with the points as the line length'
        )

    parser.add_argument('--subs', nargs='+',
                        help='The names of the subreddits to plot.')
    args = parser.parse_args()

    data = read_data(include_subs=args.subs)
    calendardir = os.path.join('plots', 'calendar')
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    if not os.path.isdir(calendardir):
        os.mkdir(calendardir)

    print('Generating Calendar plots...')
    for sub in data:
        for type in data[sub]:
            print(f'\t{sub} -- {type}')
            df = data[sub][type]
            df = df.fillna(0)

            df['datetime'] = df['created_utc'].apply(datetime.datetime.fromtimestamp)
            dmin, dmax = df['datetime'].aggregate(['min', 'max'])
            dmin, dmax = dmin.replace(hour=0, minute=0, second=0), dmax.replace(hour=0, minute=0, second=0)

            groups = {}
            # increse dmin until it is dmax and add groups to `groups` with the date as the key
            while dmin < dmax:
                groups[dmin] = df[(dmin < df['datetime']) & (df['datetime'] < dmin + delta)]
                dmin += delta
            sorted_days = list(sorted(groups.keys()))
            month = []
            # first week
            row = []
            for _ in range(sorted_days[0].weekday()):
                row.append(None)
            # days we have data for
            for day in sorted_days:
                dow = days[day.weekday()]
                if dow == 'Mon':
                    month.append(row)
                    row = []
                row.append((day.day, groups[day]))
            if len(row) != 0:
                while len(row) < 7:
                    row.append(None)
                month.append(row)


            # Plot =====================================================
            scale = 6
            width = 7
            height = len(month)
            fig = plt.figure(figsize=(scale * width, scale * height))

            x_values = np.linspace(0, 1, width + 1)
            y_values = np.linspace(0, 1, height + 1)[::-1]
            maxY = df['score'].abs().max()
            xoffset = (1/width - 1/(width+2)) / 2
            yoffset = (1/height - 1/(height+2)) / 2

            for i, y in enumerate(y_values[1:]):
                for j, x in enumerate(x_values[:-1]):
                    ax = fig.add_axes([x+xoffset, y+yoffset, 1/(width+2), 1/(height+3)], projection='polar')
                    if i == 0:
                        ax.set_title(days[j], fontsize=26, pad=26, fontweight='heavy')
                    if month[i][j] is None:
                        ax.set_axis_off()
                        continue
                    dnum, mdf = month[i][j]
                    mdf = mdf[mdf['score'].astype(int).abs() > 10]
                    fig.text(x+xoffset, y + 0.9/(height+2) + yoffset, f'{dnum}', fontsize=24, fontweight='heavy')
                    ax.spines['polar'].set_visible(False)
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    ax.set_xticks(hours)
                    ax.set_xticklabels([f'{i}' for i in range(0, 24, 3)])
                    ax.set_yticks([])
                    ax.set_ylim(0, np.log1p(maxY) * 1.1)

                    X = np.array(mdf['datetime'].apply(time_to_angle).to_list())[:, np.newaxis]
                    Y = np.array(mdf['score'].to_list())[:, np.newaxis]
                    P0 = np.concatenate([X, np.zeros([Y.shape[0], 1])], axis=1)[:, np.newaxis]
                    P1 = np.concatenate([X, np.log1p(np.abs(Y))], axis=1)[:, np.newaxis]
                    lines = np.concatenate([P0, P1], axis=1)
                    cols = get_colors(Y)

                    poslines = lines[Y[:, 0] > 0]
                    poscols = cols[Y[:, 0] > 0]
                    neglines = lines[Y[:, 0] < 0]
                    negcols = cols[Y[:, 0] < 0]

                    plc = mpl.collections.LineCollection(poslines, colors=poscols, alpha=0.33, linewidth=3, zorder=0, capstyle='round')
                    ax.add_artist(plc)
                    nlc = mpl.collections.LineCollection(neglines, colors=negcols, alpha=0.75, linewidth=3, zorder=1, capstyle='round')
                    ax.add_artist(nlc)

                    cut = min(3, len(P1))
                    if cut == 0:  # top N scatter below ONLY ============================================
                        continue
                    top = np.array(sorted(P1, key=lambda k: k[0][1], reverse=True))[:cut]
                    ax.scatter(top[:, :, 0], top[:, :, 1], c='#90be6d', zorder=11, s=50)
                    bottom = np.array(sorted(P1, key=lambda k: k[0][1]))[:cut]
                    ax.scatter(bottom[:, :, 0], bottom[:, :, 1], c='#f94144', zorder=10, s=50)
            fig.tight_layout(pad=1.15)
            fig.savefig(os.path.join(calendardir, f'calendar_{sub}_{type}.png'), bbok_inches='tight', dpi=128)
