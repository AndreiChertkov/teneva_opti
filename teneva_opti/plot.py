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
              xlabel='Number of requests', ylabel=None, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    for i, (name, item) in enumerate(data.items()):
        if item.get('skip') == True:
            continue

        y = item['mean']
        y_best = np.array(item['best'])
        y_wrst = np.array(item['wrst'])
        x = np.arange(len(y))
        linewidth = 3 if name_spec == name else 2
        ax.plot(x, y, label=name, color=colors[i],
            marker='o', markersize=4, linewidth=linewidth)

        ax.fill_between(x, y_wrst, y_best, alpha=0.5, color=colors[i])

    _prep_ax(ax, xlog=True, ylog=False, leg=True)
    # ax.set_xlim(m_min, 2.E+4)
    # ax.set_ylim(plot_opts[bm]['y_min'], plot_opts[bm]['y_max'])

    #yticks = [1.8E+3, 2.0E+3, 2.2E+3, 2.4E+3, 2.6E+3, 2.8E+3, 3.0E+3, 3.2E+3]
    #ax.set(yticks=yticks, yticklabels=[int(])
    #ax.get_yaxis().get_major_formatter().labelOnlyBase = False

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def _prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()
        ax.set_yscale('symlog')

    if leg:
        ax.legend(loc='upper right', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)