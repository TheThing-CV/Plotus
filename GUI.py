# importing libraries
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multicomp as multi
from scipy.stats.mstats import normaltest
from scipy.stats import f, stats, pearsonr
from numpy import mean
import missingno
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit

sns.set_style('white')  # white, whitegrid, dark, darkgrid
sns.set_context('notebook')

# creating directories if needed
if os.path.exists('../../Plots'):
    pass
else:
    os.makedirs('../../Plots/')

if os.path.exists('../../Analysis'):
    pass
else:
    os.makedirs('../../Analysis/')


# main function for data analysis
def analyze_us():
    global data
    if data is None:
        return

    method = combo_analysis.get()
    if method == 'Descriptive stats':
        describe()
    elif method == 'Normality test':
        normality()
    elif method == 'Levene\'s test':
        homogeneity()
    elif method == 'ANOVA one-way':
        anova_analysis()
    elif method == 'Correlation':
        correlation_analysis()
    elif method == 'Linear regression':
        linear_reg()
    elif method == 'Logistic regression':
        logistic_reg()
    else:
        print_status('Warning: Choose analysis', 'red')


# print status of the program
def print_status(text, color):
    message.config(text=text, foreground=color)


# descriptive analysis
def describe():
    global data

    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    continuous_columns = [col for col in data.columns if data[col].dtype != 'object']

    formula = var_formula.get()
    writer = pd.ExcelWriter('../../Analysis/descriptive statistics.xlsx')
    col_number = 1
    row_number = 1

    if formula == '':
        desc_data = data.describe()
        desc_data.to_excel(writer, sheet_name='Sheet1', startcol=1)

        for col in categorical_columns:
            data[col].value_counts().to_frame().to_excel(writer, sheet_name='Sheet1', startrow=11, startcol=col_number)
            col_number += 3

    else:
        x_list = formula.split('~')[0].split('+')
        y = None
        try:
            y = formula.split('~')[1]
        except:
            pass
        for x in x_list:
            if x not in data.columns:
                print_status("Warning: No such continuous column.", 'red')
                return
            if y is not None and y not in data.columns:
                print_status('Warning: No such categorical column.', 'red')
                return
            if y is not None and data[y].dtype != 'object':
                print_status('Warning: ~column has to be categorical.', 'red')
                return
            if y is None:
                desc_data = data[x].describe().to_frame()
                desc_data.to_excel(writer, sheet_name='Sheet1', startcol=col_number, startrow=row_number)
                col_number += 3
            elif y is not None and data[x].dtype != 'object':
                group_data = data.groupby(y)[x].describe().to_frame()
                group_data.to_excel(writer, sheet_name='Sheet1', startcol=col_number, startrow=row_number)
                col_number += 4
            else:
                group_data = data.groupby(y)[x].value_counts().to_frame()
                group_data.to_excel(writer, sheet_name='Sheet1', startcol=col_number, startrow=row_number)
                col_number += 4

    writer.save()
    os.startfile('../../Analysis\descriptive statistics.xlsx')


# checking for normality
def normality():
    data_dropped_na = data.dropna()
    column_name = "D’Agostino and Pearson’s Normality Test"

    formula = var_formula.get()
    if formula == '':
        print_status('Warning: Please, specify column names in formula.', 'red')
        return
    x_list = formula.split('~')[0].split('+')
    y = None
    try:
        y = formula.split('~')[1]
    except:
        pass

    test_list = []
    p_value_list = []
    index_list = []

    for x in x_list:
        if x not in data_dropped_na.columns:
            print_status("Warning: No such continuous column.", 'red')
            return
        if y is not None and y not in data_dropped_na.columns:
            print_status('Warning: No such categorical column.', 'red')
            return
        if y is None:
            test, p_value = normaltest(data_dropped_na[x])
            test_list.append(test)
            p_value_list.append(p_value)
            index_list.append(x)
        else:
            for i in set(data_dropped_na[y]):
                test, p_value = normaltest(data_dropped_na[data_dropped_na[y] == i][x])

                test_list.append(test)
                p_value_list.append(p_value)
                index_list.append(x + '[' + i + ']')

    df = pd.DataFrame({column_name: test_list, "p Value": p_value_list}, index=index_list)
    writer = pd.ExcelWriter('../../Analysis/Normality.xlsx')
    df.to_excel(writer, sheet_name='Sheet1', startcol=1)
    # df.to_excel(writer, sheet_name='Sheet1', startcol=7)
    writer.save()
    print_status('Status: Normality test performed', 'black')
    os.startfile('../../Analysis\\Normality.xlsx')


# checking for homogeneity
def homogeneity():
    data_dropped_na = data.dropna()

    formula = var_formula.get()
    if formula == '':
        print_status('Warning: Please, specify column names in formula.', 'red')
        return

    x_list = formula.split('~')[0].split('+')
    y = None
    try:
        y = formula.split('~')[1]
    except:
        pass

    if y is None or data_dropped_na[y].dtype != 'object':
        print_status('Warning: No categorical column selected', 'red')
        return

    test_list = []
    p_value_list = []
    index_list = []

    for x in x_list:
        if x not in data_dropped_na.columns:
            print_status("Warning: No such continuous column.", 'red')
            return
        if y is not None and y not in data_dropped_na.columns:
            print_status('Warning: No such categorical column.', 'red')
            return

        series_list = []

        for i in set(data_dropped_na[y]):
            series_list.append(data_dropped_na[data_dropped_na[y] == i][x])

        test_list.append(our_levene(series_list)[0])
        p_value_list.append(our_levene(series_list)[1])
        index_list.append(x)

    df = pd.DataFrame({"Levene's W": test_list, "p Value": p_value_list}, index=index_list)
    writer = pd.ExcelWriter('../../Analysis/Homogeneity.xlsx')
    df.to_excel(writer, sheet_name='Sheet1', startcol=1)
    # df.to_excel(writer, sheet_name='Sheet1', startcol=7)
    writer.save()
    print_status('Status: Levene\'s test performed', 'black')
    os.startfile('../../Analysis\Homogeneity.xlsx')


# custom Levine test
def our_levene(lists):
    dev_from_group_mean = []
    group_dev_mean = []
    group_size = []
    for i in lists:
        dev_from_group_mean.append(abs(i - i.mean()))
        group_dev_mean.append((abs(i - i.mean())).mean())
        group_size.append(len(i))

    grand_mean = mean(group_dev_mean)
    dev_from_grand_mean = []
    sums_of_dev_from_grand_mean = []

    for i in dev_from_group_mean:
        dev_from_grand_mean.append((i - grand_mean) ** 2)
        sums_of_dev_from_grand_mean.append(((i - grand_mean) ** 2).sum())

    sum_of_dev_from_grand_mean = sum(sums_of_dev_from_grand_mean)

    effect = 0
    for i in range(len(group_size)):
        effect += (group_dev_mean[i] - grand_mean) ** 2 * group_size[i]

    error = sum_of_dev_from_grand_mean - effect

    df_effect = len(group_size) - 1
    df_error = sum(group_size) - len(group_size)

    mean_square_effect = effect / df_effect
    mean_square_error = error / df_error

    f_effect = mean_square_effect / mean_square_error

    p_value = f.sf(f_effect, df_effect, df_error)

    return f_effect, p_value


# analysis of variance
def anova_analysis():
    if var_formula.get() == '':
        print_status("Warning: Formula is missing", 'red')
        return

    data_dropped_na = data.dropna()
    continuous_columns = [col for col in data_dropped_na.columns if data_dropped_na[col].dtype != 'object']

    col = 1
    writer = pd.ExcelWriter('../../Analysis/ANOVA.xlsx')

    if '.' in var_formula.get():
        dependent_vars = continuous_columns
    else:
        dependent_vars = var_formula.get().split('~')[0].split('+')

    for dependent in dependent_vars:

        mc1 = multi.MultiComparison(data_dropped_na[dependent], data_dropped_na[var_formula.get().split('~')[1]])
        result = mc1.tukeyhsd()
        t = result.summary().as_text()
        a_list = t.split('\n')
        cols = [col for col in a_list[2].split(' ') if col]
        df = pd.DataFrame(columns=cols)
        for i in range(4, len(a_list) - 1):
            items = [item for item in a_list[i].split(' ') if item]
            df.loc[i - 4] = items

        formula = dependent + '~' + var_formula.get().split('~')[1]
        mod = ols(formula, data=data_dropped_na).fit()

        aov_table = sm.stats.anova_lm(mod, typ=2)

        caption = pd.DataFrame(columns=[dependent])
        caption.to_excel(writer, sheet_name='Sheet1', startrow=2, startcol=col)
        aov_table.to_excel(writer, sheet_name='Sheet1', startcol=col, startrow=3)
        df.to_excel(writer, sheet_name='Sheet1', startrow=3, startcol=col + 7)
        col += 16
    writer.save()

    os.startfile('../../Analysis\ANOVA.xlsx')
    print_status('Status: Successful analysis', 'black')


def correlation_analysis():

    data_dropped_na = data.dropna()
    continuous_columns = [col for col in data_dropped_na.columns if data_dropped_na[col].dtype != 'object']

    writer = pd.ExcelWriter('../../Analysis/Correlation.xlsx')

    r_results = data_dropped_na.corr()
    r_results.to_excel(writer, sheet_name="Pearson's r", startrow=2, startcol=2)
    p_results = calculate_pvalues(data_dropped_na)
    p_results.to_excel(writer, sheet_name="P Values", startrow=2, startcol=2)
    writer.save()

    os.startfile('../../Analysis\Correlation.xlsx')
    print_status('Status: Successful analysis', 'black')


def linear_reg():
    if var_formula.get() == '':
        print_status("Warning: Formula is missing", 'red')
        return

    data_dropped_na = data.dropna()
    continuous_columns = [col for col in data_dropped_na.columns if data_dropped_na[col].dtype != 'object']

    col = 1
    writer = pd.ExcelWriter('../../Analysis/Linear_reg.xlsx')

    dependent_var = var_formula.get().split('~')[0]
    regressor = var_formula.get().split('~')[1]

    formula = dependent_var + '~' + regressor
    mod = ols(formula, data=data_dropped_na).fit()

    caption = pd.DataFrame(columns=['Linear Regression'])
    caption.to_excel(writer, sheet_name='Sheet1', startrow=2, startcol=col)

    df = pd.concat((mod.params, mod.tvalues, mod.pvalues), axis=1)

    # print(mod.conf_int(alpha=0.05, cols=None))
    df.rename(columns={0: 'beta', 1: 't', 2: 'p_value'}).to_excel(writer, sheet_name='Sheet1', startrow=4,
                                                                  startcol=col)
    col += 6

    writer.save()

    os.startfile('../../Analysis\Linear_reg.xlsx')
    print_status('Status: Successful analysis', 'black')


def logistic_reg():
    if var_formula.get() == '':
        print_status("Warning: Formula is missing", 'red')
        return

    data_dropped_na = data.dropna()
    continuous_columns = [col for col in data_dropped_na.columns if data_dropped_na[col].dtype != 'object']

    col = 1
    writer = pd.ExcelWriter('../../Analysis/Logistic_reg.xlsx')

    dependent_var = var_formula.get().split('~')[0]
    regressor = var_formula.get().split('~')[1]

    data_dropped_na['Category_log'] = data_dropped_na[dependent_var].map({data[dependent_var].unique()[0]: 1,
                                                                      data[dependent_var].unique()[1]: 0})

    formula = 'Category_log' + '~' + regressor
    mod = logit(formula, data=data_dropped_na).fit()

    caption = pd.DataFrame(columns=['Logistic Regression'])
    caption.to_excel(writer, sheet_name='Sheet1', startrow=2, startcol=col)

    df = pd.concat((mod.params, mod.bse, mod.pvalues), axis=1)

    # print(mod.conf_int(alpha=0.05, cols=None))
    df.rename(columns={0: 'coeff', 1: 'std error', 2: 'p_value'}).to_excel(writer, sheet_name='Sheet1', startrow=4,
              startcol=col)
    col += 6

    writer.save()

    os.startfile('../../Analysis\Logistic_reg.xlsx')
    print_status('Status: Successful analysis', 'black')


def calculate_pvalues(df):
    df = df._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = pearsonr(df[r], df[c])[1]
    return pvalues


# load data from disk
def load():
    global data
    file = filedialog.askopenfile(parent=root, mode='rb', title='Choose your file',
                                  filetypes=(("Excel files and .csv", "*.xlsx;*.xls;*.csv")
                                             , ("All files", "*.*")))

    if file is not None:

        if file.name.split('.')[-1] == 'csv':
            data = pd.read_csv(file)
        elif file.name.split('.')[-1] == 'xlsx' or file.name.split('.')[-1] == 'xls':
            data = pd.read_excel(file)
        else:
            print_status("Warning: Unsupported file type", 'red')
            return
        print_status("Status: Data are successfully loaded", 'black')
        combo_x.config(values=data.columns.tolist())
        combo_x.set(data.columns[0])
        combo_y.config(values=data.columns.tolist())
        combo_y.set(data.columns[0])
        values_list = ['None'] + data.columns.tolist()
        combo_by.config(values=values_list)
        combo_by.set(values_list[0])
        types_list = ['Histogram', 'Pair plot', 'Scatter plot', 'Bar plot', 'Count bar', 'Box plot', 'Violin plot',
                      'Beeswarm plot', 'Correlation matrix', 'Cluster plot', 'Missing data with matrix',
                      'Missing data with bars', 'Missing data correlations']
        type_combo.config(values=types_list)
        type_combo.set(types_list[0])
        analysis_types = ['None', 'Descriptive stats', 'Normality test', 'Levene\'s equality of variance',
                          'ANOVA one-way', 'Correlation', 'Linear regression', 'Logistic regression']
        combo_analysis.config(values=analysis_types)
        combo_analysis.set(analysis_types[0])
        palettes = ['Blues', 'coolwarm', 'GnBu_d', 'pastel', 'Set1',
                    'summer', 'muted', 'Spectral', 'husl', 'copper', 'magma']
        combo_palette.config(values=palettes)
        combo_palette.set(palettes[0])

        listbox.delete(0, END)
        for item in data.columns:
            # insert each new item to the end of the listbox
            listbox.insert('end', item)



# core plotting procedures
def plot_us():
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    by = combo_by.get()
    if by == 'None':
        by = None

    data_dropped_na = data.dropna()

    plot_type = type_combo.get()

    if plot_type == 'Histogram':

        g = sns.distplot(data_dropped_na[combo_x.get()], rug=True, rug_kws={'color': '#777777', 'alpha': 0.2},
                         hist_kws={'edgecolor': 'black', 'color': '#6899e8', 'label': 'розподіл'},
                         kde_kws={'color': 'black', 'alpha': 0.2, 'label': 'ядрова оцінка густини'})
        sns.despine(left=True, bottom=True)  # видалити осі повністю
        g.set_xlabel(combo_x.get(), color='black', fontsize=15, alpha=0.5)
        g.set_ylabel('Густина', color='black', fontsize=15, alpha=0.5)
        if by is not None:
            plt.legend(loc='upper right')

        fig.savefig('../../Plots/hist.pdf')
        plt.close(fig)
        os.startfile('../../Plots\hist.pdf')
        return

    if plot_type == 'Pair plot':
        sns_plot = sns.pairplot(data_dropped_na, hue=by, palette=combo_palette.get())

        sns_plot.savefig('../../Plots/pairplot.pdf')
        plt.close(fig)
        os.startfile('../../Plots\pairplot.pdf')
        return

    if plot_type == 'Scatter plot':
        a = sns.jointplot(combo_x.get(), combo_y.get(), data=data_dropped_na, kind='reg', color='#5394d6',

                          marginal_kws={'rug': True, 'bins': 25, 'hist_kws': {'edgecolor': 'black'}},
                          joint_kws={'scatter_kws': {'alpha': 0.7}})
        plt.setp(a.ax_marg_x.patches, linewidth=1.0, color='#a9c8e8')
        plt.setp(a.ax_marg_y.patches, linewidth=1.0, color='#a9c8e8')
        a.ax_joint.set_xlabel(combo_x.get(), fontsize=15, alpha=0.7)
        a.ax_joint.set_ylabel(combo_y.get(), fontsize=15, alpha=0.7)
        a.annotate(stats.pearsonr)
        plt.savefig('../../Plots/scatter.pdf')
        plt.close()
        os.startfile('../../Plots\scatter.pdf')

        return

    if plot_type == 'Bar plot':

        ax = sns.barplot(x=combo_x.get(), y=combo_y.get(), hue=by, data=data_dropped_na, palette=combo_palette.get(),
                         errcolor='0.4', errwidth=1.1)
        ax.set_ylabel('Середнє значення ' + combo_y.get(), color='#666666')
        ax.set_xlabel(combo_x.get(), color='#666666')
        if by is not None:
            plt.legend(loc=[0.8, 0.9])
        sns.despine()
        fig.savefig('../../Plots/barplot.pdf')
        plt.close(fig)
        os.startfile('../../Plots\\barplot.pdf')
        return

    if plot_type == 'Count bar':
        ax = sns.countplot(x=combo_x.get(), hue=by, data=data_dropped_na, palette=combo_palette.get())
        ax.set_ylabel('Кількість', color='#666666')
        ax.set_xlabel(var_x.get(), color='#666666')
        if by is not None:
            plt.legend(loc=[0.8, 0.9])
        sns.despine()
        fig.savefig('../../Plots/countbar.pdf')
        plt.close(fig)
        os.startfile('../../Plots\\countbar.pdf')
        return

    if plot_type == 'Box plot':

        ax = sns.boxplot(combo_x.get(), combo_y.get(), data=data_dropped_na, hue=by, width=0.4,
                         palette=combo_palette.get())
        ax.set_ylabel(combo_y.get(), color='#666666')
        ax.set_xlabel(combo_x.get(), color='#666666')
        if by is not None:
            plt.legend(loc='upper right')
        sns.despine()
        plt.savefig('../../Plots/Boxplot.pdf')
        plt.close(fig)
        os.startfile('../../Plots\Boxplot.pdf')
        return

    if plot_type == 'Violin plot':

        ax = sns.violinplot(combo_x.get(), combo_y.get(), data=data_dropped_na, hue=by, scale='count', split=True,
                            palette=combo_palette.get())
        ax.set_ylabel(combo_y.get(), color='#666666')
        ax.set_xlabel(combo_x.get(), color='#666666')
        if by is not None:
            plt.legend(loc='upper right')
        sns.despine()
        plt.savefig('../../Plots/violin.pdf')
        plt.close(fig)
        os.startfile('../../Plots\\violin.pdf')
        return

    if plot_type == 'Beeswarm plot':
        ax = sns.swarmplot(combo_x.get(), combo_y.get(), data=data_dropped_na, hue=by, alpha=0.7,
                           palette=combo_palette.get())

        mean_width = .5

        for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
            sample_name = text.get_text()

            mean_val = data_dropped_na[data_dropped_na[combo_x.get()] == sample_name][combo_y.get()].mean()

            ax.plot([tick - mean_width / 2, tick + mean_width / 2], [mean_val, mean_val], lw=2, color='#777777')

        ax.set_ylabel(combo_y.get(), color='#666666')
        ax.set_xlabel(combo_x.get(), color='#666666')
        sns.despine()
        plt.savefig('../../Plots/beeswarm.pdf')
        plt.close(fig)
        os.startfile('../../Plots\\beeswarm.pdf')
        return

    if plot_type == 'Correlation matrix':
        corr = data_dropped_na.corr()
        colormap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", linewidths=0.5)
        plt.savefig('../../Plots/correlation_m.pdf', bbox_inches='tight')
        plt.close(fig)
        os.startfile('../../Plots\\correlation_m.pdf')
        return

    if plot_type == 'Cluster plot':
        sns.clustermap(data_dropped_na.corr(), metric="correlation", method="single", cmap='vlag', linewidths=0.5,
                       figsize=(20, 12))
        plt.savefig('../../Plots/cluster_plot.pdf', bbox_inches='tight')
        plt.close(fig)
        os.startfile('../../Plots\\cluster_plot.pdf')
        return

    if plot_type == 'Missing data with matrix':
        figsize = None
        if len(data.columns) > 10:
            figsize = (30, 27)
        else:
            figsize = (25, 10)

        ax = missingno.matrix(data if len(data) < 500 else data.sample(500), inline=False, figsize=figsize)

        plt.savefig('../../Plots/missing matrix.pdf')
        plt.close(fig)
        os.startfile('../../Plots\\missing matrix.pdf')
        return

    if plot_type == 'Missing data with bars':
        figsize = None
        if len(data.columns) > 10:
            figsize = (30, 27)
        else:
            figsize = (25, 10)

        ax = missingno.bar(data if len(data) < 500 else data.sample(500), inline=False, figsize=figsize)

        plt.savefig('../../Plots/missing bars.pdf')
        plt.close(fig)
        os.startfile('../../Plots\\missing bars.pdf')
        return

    if plot_type == 'Missing data correlations':
        ax = missingno.heatmap(data, inline=False, figsize=(25, 25))

        plt.savefig('../../Plots/missing correlations.pdf')
        plt.close(fig)
        os.startfile('../../Plots\\missing correlations.pdf')
        return


def on_select(evt):
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    var_formula.set(var_formula.get() + value)
    formula_analysis.delete(0, END)
    formula_analysis.insert(0, var_formula.get())


def add_plus():
    var_formula.set(var_formula.get() + '+')
    formula_analysis.delete(0, END)
    formula_analysis.insert(0, var_formula.get())


def add_tilda():
    var_formula.set(var_formula.get() + '~')
    formula_analysis.delete(0, END)
    formula_analysis.insert(0, var_formula.get())


def add_dot():
    formula_analysis.delete(0, END)
    var_formula.set('.~')
    formula_analysis.insert(0, var_formula.get())


def delete_formula():
    var_formula.set('')
    formula_analysis.delete(0, END)


# main GUI setup
fig, ax = plt.subplots(1, 1)
data = None

root = Tk()
root.resizable(0, 0)

root.wm_title("Plotus")
ttk.Button(root, text="Load data", command=load, width=40).grid(row=0, column=0, columnspan=3, pady=5, padx=5)

type_value = StringVar()
ttk.Label(root, text='Choose plot type: ').grid(row=1, column=0, sticky='W', padx=7)
type_combo = ttk.Combobox(root, textvariable=type_value)
type_combo.grid(row=1, column=1, padx=7)

ttk.Label(root, text='Choose x-axis: ').grid(row=2, column=0, sticky='W', padx=7)
var_x = StringVar()
combo_x = ttk.Combobox(root, textvariable=var_x)
combo_x.grid(row=2, column=1, padx=7)

var_y = StringVar()
ttk.Label(root, text='Choose y-axis: ').grid(row=3, column=0, sticky='W', padx=7)
combo_y = ttk.Combobox(root, textvariable=var_y)
combo_y.grid(row=3, column=1, padx=7)

var_by = StringVar()
ttk.Label(root, text='Choose \'by\' factor: ').grid(row=4, column=0, sticky='W', padx=7)
combo_by = ttk.Combobox(root, textvariable=var_by)
combo_by.grid(row=4, column=1, padx=7)

ttk.Label(root, text='Choose palette:').grid(row=5, column=0, sticky='W', padx=7)
var_palette = StringVar()
combo_palette = ttk.Combobox(root, textvariable=var_palette)
combo_palette.grid(row=5, column=1, padx=7)

ttk.Button(root, text='Plot', command=plot_us, width=40).grid(row=6, column=0, columnspan=3, padx=5, pady=5)

listbox = Listbox(root, width=15, height=5)

listbox.grid(row=7, column=0, rowspan=3, padx=5, pady=5)
# create a vertical scrollbar to the right of the listbox
y_scroll = Scrollbar(root, command=listbox.yview, orient=VERTICAL)
y_scroll.grid(row=7, column=0, rowspan=3, sticky='nse')
listbox.configure(yscrollcommand=y_scroll.set)

listbox.bind('<<ListboxSelect>>', on_select)

ttk.Button(root, text='+', command=add_plus, width=2).grid(row=10, column=0, padx=14, pady=5, columnspan=1, sticky='W')
ttk.Button(root, text='.', command=add_dot, width=2).place(x=38, y=274)
ttk.Button(root, text='~', command=add_tilda, width=2).place(x=62, y=274)
ttk.Button(root, text='D', command=delete_formula, width=2).place(x=86, y=274)


analysis_label = ttk.Label(root, text="Choose analysis:")
analysis_label.grid(row=7, column=1, sticky='WS', padx=5)
analysis_label.config(font=('default', 8))
var_analysis = StringVar()
combo_analysis = ttk.Combobox(root, textvariable=var_analysis)
combo_analysis.grid(row=8, column=1, padx=7)

formula_label = ttk.Label(root, text='Write formula:')
formula_label.grid(row=9, column=1, sticky='WS', padx=5)
formula_label.config(font=('default', 8))
var_formula = StringVar()
formula_analysis = ttk.Entry(root, textvariable=var_formula, width=23)
formula_analysis.grid(row=10, column=1, padx=7)

analyze_button = ttk.Button(root, text='Analyze', command=analyze_us, width=40)
analyze_button.grid(row=11, column=0, columnspan=3, padx=5, pady=5)

message = ttk.Label(root, text='Status: Ready')
message.grid(row=12, column=0, columnspan=2, pady=2)
message.config(font=('default', 7))

# run module

if __name__ == '__main__':
    root.mainloop()
