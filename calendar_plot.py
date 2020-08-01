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
    if Y.shape[0] == 0:
        return np.zeros_like(Y)
    red = np.array(mpl.colors.to_rgba(colors[-1]))
    gre = np.array(mpl.colors.to_rgba(colors[1]))
    cols = np.zeros([Y.shape[0], 4])
    cols[(Y > 0)[:, 0]] = gre
    cols[(Y < 0)[:, 0]] = red
    cols[:, -1] = 1 - 66/np.max([np.sqrt(np.abs(Y[:, 0])), np.ones_like(Y[:, 0])*75])
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

    if len(args.subs) == 0:
        print('Please provide an argument to --subs.')
        exit(1)
    usecols = ['created_utc', 'score']
    data = read_data(include_subs=args.subs, submission_cols=usecols, comment_cols=usecols)
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

            lax = None
            for i, y in enumerate(y_values[1:]):
                for j, x in enumerate(x_values[:-1]):
                    ax = fig.add_axes([x+xoffset, y+yoffset, 1/(width+2), 1/(height+3)], projection='polar')
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    ax.set_xticks(hours)
                    ax.set_xticklabels([f'{i}' for i in range(0, 24, 3)])
                    ax.set_yticks([])
                    ax.set_ylim(0, np.log1p(maxY) * 1.2)
                    if i == 0:
                        ax.set_title(days[j], fontsize=26, pad=26, fontweight='heavy')
                    if month[i][j] is None:
                        if lax is None:
                            lax = ax
                        else:
                            ax.set_axis_off()
                        continue
                    dnum, mdf = month[i][j]
                    mdf = mdf[mdf['score'].astype(int).abs() > 10]
                    fig.text(x+xoffset, y + 0.9/(height+2) + yoffset, f'{dnum}', fontsize=24, fontweight='heavy')
                    ax.spines['polar'].set_visible(False)
                    if len(mdf) == 0:
                        continue

                    X = np.array(mdf['datetime'].apply(time_to_angle).to_list())[:, np.newaxis]
                    Y = np.array(mdf['score'].to_list())[:, np.newaxis]
                    P0 = np.concatenate([X, np.zeros([Y.shape[0], 1])], axis=1)[:, np.newaxis]
                    P1 = np.concatenate([X, np.log1p(np.abs(Y))], axis=1)[:, np.newaxis]
                    P1_ = np.concatenate([X, np.sign(Y) * np.log1p(np.abs(Y))], axis=1)[:, np.newaxis]
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

                    cut = min(1, len(P1_))
                    if cut == 0:  # top N scatter below ONLY =====================================
                        continue
                    htol = np.array(sorted(P1_, key=lambda k: k[0][1], reverse=True))
                    ax.scatter(htol[:cut, :, 0], np.abs(htol[:cut, :, 1]), c='#90be6d', zorder=11, s=50)
                    ax.scatter(htol[-cut:, :, 0], np.abs(htol[-cut:, :, 1]), c='#f94144', zorder=10, s=50)

            # Draw legend
            lax.set_axis_off()
            Y = np.arange(1, np.math.log(maxY, 10)+1, 1)[:, np.newaxis]
            Y = np.concatenate([10**Y, -1*10**Y[::-1]])
            X = np.linspace(0.25*np.pi, 0.75*np.pi, Y.shape[0])[:, np.newaxis]
            P0 = np.concatenate([X, np.zeros([Y.shape[0], 1])], axis=1)[:, np.newaxis]
            P1 = np.concatenate([X, Y], axis=1)[:, np.newaxis]
            P1_ = np.concatenate([X, np.log1p(np.abs(Y))], axis=1)[:, np.newaxis]
            lines = np.concatenate([P0, P1_], axis=1)
            cols = get_colors(Y)


            poslines = lines[Y[:, 0] > 0]
            poscols = cols[Y[:, 0] > 0]
            neglines = lines[Y[:, 0] < 0]
            negcols = cols[Y[:, 0] < 0]

            plc = mpl.collections.LineCollection(poslines, colors=poscols, alpha=0.33, linewidth=3, zorder=0, capstyle='round')
            lax.add_artist(plc)
            nlc = mpl.collections.LineCollection(neglines, colors=negcols, alpha=0.75, linewidth=3, zorder=1, capstyle='round')
            lax.add_artist(nlc)
            for x, y in P1[:, 0, :]:
                e = int(np.math.log(np.abs(y+1), 10))
                s = int(np.sign(y))
                plot_y = np.log1p(np.abs(y))
                lax.text(x, plot_y,
                    f'${int(s*10)}^{e}$',
                    fontsize=16, fontweight='semibold', ha='center', va='center'
                    )
            lax.scatter([np.pi * 3/2, np.pi * 3/2], [3, 4], c=colors[:0:-1])

            # Save figure
            print('\tSaving')
            # fig.tight_layout(pad=1.15)
            fig.savefig(os.path.join(calendardir, f'calendar_{sub}_{type}.png'), bbok_inches='tight', dpi=128)
