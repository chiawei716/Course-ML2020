import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, log
from scipy.stats import pearsonr, spearmanr, kendalltau

TICK = 0.25
CMAP = {'Iris-setosa': 'red', 'Iris-versicolor': 'limegreen', 'Iris-virginica': 'blue'}


def Sturge(n):
    return round(1 + 3.322 * log(n))


def btm_boundary(y):
    return floor((y / TICK) - 0.5) * 0.25


def top_boundary(y):
    return (ceil(y / TICK) + 0.5) * 0.25


def draw_hist(axes, bins, data, x_min, x_max, text, fontsize=12):
    n, b, p = axes.hist(
        x=data,
        bins=bins,
        facecolor='cyan',
        edgecolor='black',
        alpha=1
    )
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(0, n.max() * 1.3)
    axes.text(
        x=(x_min + x_max) * 0.5,
        y=n.max() * 1.1,
        s=text,
        horizontalalignment='center',
        fontsize=fontsize
    )
    return


def draw_scatter(axes, x, y, c, x_min, x_max, x_pos, y_pos):
    axes.scatter(
        x=x,
        y=y,
        s=15,
        c=c
    )
    axes.set_xlim(x_min, x_max)
    if x_pos:
        axes.xaxis.set_ticks_position(x_pos)
    else:
        axes.set_xticks([])
    if y_pos:
        axes.yaxis.set_ticks_position(y_pos)
    else:
        axes.set_yticks([])
    return

def draw_correlation(axes, x, y, fontsize=9):
    pearson_r, pearson_p = pearsonr(x, y)
    kendall_r, kendall_p = kendalltau(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    axes.text(
        x=0.5,
        y=0.5,
        s=f'Pearson: {pearson_r:.2f}\nSpearman: {spearman_r:.2f}\nKendall: {kendall_r:.2f}\n\n- chwchao',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=fontsize
    )
    axes.set_xticks([])
    axes.set_yticks([])


# Read dataset
df = pd.read_csv('IRIS.csv')
h, w = df.shape
bins = Sturge(h)

# Extract info
sl_min = btm_boundary(df['sepal_length'].min())
sl_max = top_boundary(df['sepal_length'].max())
sw_min = btm_boundary(df['sepal_width'].min())
sw_max = top_boundary(df['sepal_width'].max())
pl_min = btm_boundary(df['petal_length'].min())
pl_max = top_boundary(df['petal_length'].max())
pw_min = btm_boundary(df['petal_width'].min())
pw_max = top_boundary(df['petal_width'].max())

# Create subplots
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

# Histograms
draw_hist(axes[0, 0], bins, df['sepal_length'], sl_min, sl_max, 'Sepal.Length')
draw_hist(axes[1, 1], bins, df['sepal_width'], sw_min, sw_max, 'Sepal.Width')
draw_hist(axes[2, 2], bins, df['petal_length'], pl_min, pl_max, 'Petal.Length')
draw_hist(axes[3, 3], bins, df['petal_width'], pw_min, pw_max, 'Petal.Width')

# Scatter
c = [CMAP[iris] for iris in df['species']]
draw_scatter(axes[0, 1], df['sepal_width'],
             df['sepal_length'], c, sw_min, sw_max, 'top', None)
draw_scatter(axes[0, 2], df['petal_length'],
             df['sepal_length'], c, pl_min, pl_max, 'top', None)
draw_scatter(axes[0, 3], df['petal_width'],
             df['sepal_length'], c, pw_min, pw_max, 'top', 'right')
draw_scatter(axes[1, 2], df['petal_length'],
             df['sepal_width'], c, pl_min, pl_max, None, None)
draw_scatter(axes[1, 3], df['petal_width'], df['sepal_width'],
             c, pw_min, pw_max, None, 'right')
draw_scatter(axes[2, 3], df['petal_width'],
             df['petal_length'], c, pw_min, pw_max, None, 'right')


# Correlation
draw_correlation(axes[1, 0], df['sepal_length'], df['sepal_width'])
draw_correlation(axes[2, 0], df['sepal_length'], df['petal_length'])
draw_correlation(axes[2, 1], df['sepal_width'], df['petal_length'])
draw_correlation(axes[3, 0], df['sepal_length'], df['petal_width'])
draw_correlation(axes[3, 1], df['sepal_width'], df['petal_width'])
draw_correlation(axes[3, 2], df['petal_length'], df['petal_width'])

plt.savefig('IrisPairwiseScatterPlot.png', dpi=100)
