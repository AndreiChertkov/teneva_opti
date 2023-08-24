import matplotlib as mpl
import numpy as np


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


def plot_deps(data, colors, fpath=None, name_spec=None,
              xlabel='Number of requests', ylabel=None, title=None,
              lim_x=None, lim_y=None, ylog=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    is_neg = False
    for i, (name, item) in enumerate(data.items()):
        if item.get('skip') == True:
            continue

        y = np.array(item['mean'])
        y_best = np.array(item['best'])
        y_wrst = np.array(item['wrst'])
        x = np.arange(len(y)) + 1
        linewidth = 1 if name_spec == name else 1
        ax.plot(x, y, label=name, color=colors[i],
            marker='o', markersize=0, linewidth=linewidth)

        ax.fill_between(x, y_wrst, y_best, alpha=0.4, color=colors[i])

        is_neg = is_neg or np.min(y_best) < 0 or np.min(y_wrst) < 0

    _prep_ax(ax, xlog=True, ylog=ylog, leg=True, is_neg=is_neg)

    if lim_x is not None:
        ax.set_xlim(*lim_x)
    if lim_y is not None:
        ax.set_ylim(*lim_y)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def _prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None,
             is_neg=False):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()
        if is_neg:
            ax.set_yscale('symlog')

    if leg:
        ax.legend(loc='upper left', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)
