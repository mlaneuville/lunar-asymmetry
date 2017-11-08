# coding: utf-8

import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
rcParams.update(params)

def plot_correlation_map(df, source, target):
    corr = df.corr()
    corr = corr.loc[source, target]
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 14}
    )

    plt.xticks(range(len(target)), target, ha="left")
    plt.setp(plt.xticks()[1], rotation=0)
    plt.savefig("img/analysis-correlations.png", format="png", bbox_inches="tight")

def make_scatter_plot(ax, var, color):
    cm = plt.cm.get_cmap('viridis')
    cs = ax.scatter(df['nearside_'+var], df['farside_'+var], c=df[color], cmap=cm)
    if var in ['mean', 'std']:
        ax.scatter(ohtake_data[var+'_ns'], ohtake_data[var+'_fs'], lw=0, c='r', marker='*', s=300)
    ax.set_xlabel("Nearside "+var+" Mg#", fontsize=20)
    ax.set_xlim(52, 60)
    ax.set_ylim(58, 66)

    ax.set_ylabel("Farside "+var+" Mg#", fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cs, cax=cax)
    cbar.ax.set_ylabel(labels[color], rotation=90, fontsize=20, labelpad=10)
    if color[:5] == 'depth':
        tlabel = np.linspace(df[color].min()/1e3, df[color].max()/1e3, 11)
        cbar.ax.set_yticklabels(["%.0f" % a for a in tlabel])

if __name__ == "__main__":

    with open('dat/data.txt') as data_file:
        data = json.load(data_file)

    df = pd.DataFrame(data)
    df['delta_crust'] = (df['farside_crust'] - df['nearside_crust'])/1e3
    df['front'] /= 1e3
    print(df.info())

    # indices which are close enough from Ohtake data
    mean_ok = np.where(np.logical_and(abs(df["nearside_mean"] - 54.4) < 1,
                                      abs(df["farside_mean"] - 63.3) < 1))[0]

    print()
    print("Indices close to observed values: %d (out of %d)" %
          (len(mean_ok), len(df)))
    if len(mean_ok) > 0:
        print("Delay of acceptable runs: %.1f +/- %.1f Ma" %
              (df["delay"].iloc[mean_ok].mean(), df["delay"].iloc[mean_ok].std()))
        print("Plag fraction of acceptable runs: %.2f +/- %.2f" %
              (df["plag"].iloc[mean_ok].mean(), df["plag"].iloc[mean_ok].std()))
        print("Farside crust size when t > delay for acceptable runs: %.1f +/- %.1f km" %
              (df["front"].iloc[mean_ok].mean(), df["front"].iloc[mean_ok].std()))
    else:
        mean_ok = df.index
    print()

    columns = ["delay"] #, "depth_min_fs", "depth_max_fs","depth_min_ns", "depth_max_ns"]
    target = ["nearside_mean", "farside_mean", "nearside_crust", "farside_crust", "delta_crust"]

    plot_correlation_map(df.iloc[mean_ok], columns, target)

    ohtake_data = {'mean_ns':54.4, 'mean_fs':63.3, 'std_ns':5, 'std_fs':5}
    labels = {"delay": "Nearside/farside crystallization delay [Ma]",
              "plag": "Anorthosite plagioclase fraction [-]",
              "tau": "Cooling asymmetry timescale [Ma]",
              "F0": "Farside heat flow [W/m/K]",
              "F1": "Initial nearside heat flow [W/m/K]"}

    f, ax = plt.subplots(figsize=(8, 8))
    make_scatter_plot(ax, 'mean', 'delay')
    plt.savefig("img/analysis-minvis-distribution-delay.jpg", dpi=300)

    f, ax = plt.subplots(figsize=(8, 8))
    make_scatter_plot(ax, 'mean', 'plag')
    plt.savefig("img/analysis-minvis-distribution-plag.jpg", dpi=300)

    cols = ['delay', 'plag', 'front', 'delta_crust'] #'nearside_mean', 'farside_mean',
    idx = np.where(df["nearside_mean"] > 40)[0]
    sns.pairplot(df[cols].iloc[mean_ok], size=4, diag_kind='kde')
    plt.savefig("img/analysis-pairplot.png", format="png", dpi=300)
