import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if not os.path.exists('Output'):
    os.makedirs('Output')


# Plot a single histogram
def plot_hist(data_frame, x, bins):
    sns.set_style('white')  # white, whitegrid, dark, darkgrid
    sns.set_context('notebook')
    g = sns.distplot(data_frame[x].dropna(), bins=bins, rug=True, rug_kws={'color': '#777777', 'alpha': 0.2},
                     hist_kws={'edgecolor': 'black', 'color': '#6899e8', 'label': 'розподіл'},
                     kde_kws={'color': 'black', 'alpha': 0.2, 'label': 'ядрова оцінка густини'})
    sns.despine(left=True, bottom=True)  # видалити осі повністю
    # g.set_title('Розподіл значень ' + x, fontsize=15, color='#555555')
    g.set_xlabel(x, color='black', fontsize=15, alpha=0.5)
    g.set_ylabel('Густина', color='black', fontsize=15, alpha=0.5)
    plt.legend(loc='upper right')

    plt.savefig('Output/hist.pdf')
# End single histogram


# Plot a scatter plot
def plot_scatter(data_frame, x, y):
    sns.set_style('white')
    a = sns.jointplot(x, y, data=data_frame, kind='reg', color='#5394d6',
                      annot_kws={'fontsize': 14, 'loc': [-0.1, 0.85]},
                      marginal_kws={'rug': True, 'bins': 25, 'hist_kws': {'edgecolor': 'black'}},
                      joint_kws={'scatter_kws': {'alpha': 0.7}})
    plt.setp(a.ax_marg_x.patches, linewidth=1.0, color='#a9c8e8');
    plt.setp(a.ax_marg_y.patches, linewidth=1.0, color='#a9c8e8');
    a.ax_joint.set_xlabel(x, fontsize=15, alpha=0.7)
    a.ax_joint.set_ylabel(y, fontsize=15, alpha=0.7)
    plt.savefig('Output/scatter.pdf')
# End scatter plot


# Plot matrix plot
def plot_matrix(data_frame, by=None):
    sns.set_style('white')
    sns.set_context('notebook', font_scale=1.5)
    g = sns.pairplot(data_frame,  hue=by, diag_kws={'edgecolor': '#555555'},
                     palette='Blues', size=4)
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=10, fontsize=15)
    g.fig.subplots_adjust(bottom=0.13)
    try:
        g._legend.remove()
    except:
        pass

    plt.savefig('Output/matrix.pdf')
# End matrix plot

# Plot bar
def plot_bar(data_frame, x, y, by=None, count=None):
    sns.set_style('white')
    sns.set_context('notebook', font_scale=1.5)
    if count is None:
        ax = sns.barplot(x=x, y=y, hue=by, data=data_frame, palette='Blues',
                         errcolor='0.4', errwidth=1.1)
        ax.set_ylabel('Середнє значення ' + y, color='#666666')
        ax.set_xlabel(x, color='#666666')
        plt.legend(loc=[0.8, 0.9])
    else:
        ax = sns.countplot(x, data=data_frame, hue=by, palette='Blues')
        ax.set_ylabel('Кількість ', color='#666666')
        ax.set_xlabel(x, color='#666666')

    sns.despine()
    plt.savefig('Output/bar.pdf')
# End bar plot

# Plot boxplot
def plot_box(data_frame, x, y, by=None):
    sns.set_style('white')
    ax = sns.boxplot(x, y, data=data_frame, hue=by, width=0.4, palette='Blues')
    ax.set_ylabel(y, color='#666666')
    ax.set_xlabel(x, color='#666666')
    sns.despine()
    plt.savefig('Output/box.pdf')
# End boxplot

# Plot violin
def plot_violin(data_frame, x, y, by=None):
    sns.set_style('white')
    ax = sns.violinplot(x, y, data=data_frame, hue=by, split=True, palette='Blues')
    ax.set_ylabel(y, color='#666666')
    ax.set_xlabel(x, color='#666666')
    plt.legend(loc='upper right')
    sns.despine()
    plt.savefig('Output/violin.pdf')
# End violin


# Plot swarm
def plot_swarm(data_frame, x, y, by=None):
    sns.set_style('white')
    sns.set_context('notebook')
    ax = sns.swarmplot(x, y, data=data_frame, hue=by, alpha=0.7)

    mean_width = .5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        sample_name = text.get_text()

        mean_val = data_frame[data_frame[x] == sample_name][y].mean()

        ax.plot([tick - mean_width / 2, tick + mean_width / 2], [mean_val, mean_val], lw=2, color='#777777')

    ax.set_ylabel(y, color='#666666')
    ax.set_xlabel(x, color='#666666')
    sns.despine()
    plt.savefig('Output/swarm.pdf')
# End swarm

# Heatmap
def plot_heatmap(data_frame):
    sns.set_style('white')
    sns.set_context('notebook')
    fm = data_frame.pivot_table(index='month', columns='year', values='passengers')
    cg = sns.heatmap(fm, linecolor='white', linewidth=1, annot=True, fmt='d')
    plt.yticks(rotation=0)
    plt.savefig('Output/heatmap.pdf');
# End heatmap

# Cluster
def plot_cluster(data_frame):
    sns.set_style('white')
    sns.set_context('notebook')
    fm = data_frame.pivot_table(index='month', columns='year', values='passengers')
    cg = sns.clustermap(fm, standard_scale=1.0, linecolor='white', linewidth=1)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    plt.savefig('Output/cluster.pdf');
# End cluster



data = sys.argv[sys.argv.index('-data') + 1]
i = data.find('.')
ext = data[i+1:]

plot_type = sys.argv[sys.argv.index('-type') + 1]
try:
    x = sys.argv[sys.argv.index('-x') + 1]
except Exception:
    x = None

try:
    y = sys.argv[sys.argv.index('-y') + 1]
except Exception:
    y = None

try:
    by = sys.argv[sys.argv.index('-by') + 1]
except Exception:
    by = None

try:
    count = sys.argv[sys.argv.index('-count')]
except Exception:
    count = None

try:
    bins = sys.argv[sys.argv.index('-bins') + 1]
except Exception:
    bins = None


if ext == 'xlsx':
    df = pd.read_excel(data)
else:
    df = pd.read_csv(data)


if plot_type == 'hist':
    plot_hist(df, x, bins)

if plot_type == 'scatter':
    plot_scatter(df, x, y)

if plot_type == 'matrix':
    plot_matrix(df, by)

if plot_type == 'bar':
    plot_bar(df, x, y, by, count)

if plot_type == 'box':
    plot_box(df, x, y, by)

if plot_type == 'violin':
    plot_violin(df, x, y, by)

if plot_type == 'swarm':
    plot_swarm(df, x, y, by)

if plot_type == 'heatmap':
    plot_heatmap(df)

if plot_type == 'cluster':
    plot_cluster(df)


# tips = sns.load_dataset('tips')
# sns.distplot(tips['total_bill'])
#
# plt.show()

# writer = pd.ExcelWriter('data.xlsx')
# tips.to_excel(writer)
# writer.save()

