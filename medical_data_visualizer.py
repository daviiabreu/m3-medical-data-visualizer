import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import the dataset
df = pd.read_csv('medical_examination.csv')

# 2: Add an overweight column
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)

# 3: Normalize data
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4: Draw catplot
def draw_cat_plot():
    # 5: Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # 6: Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # 7: Draw the catplot using seaborn's catplot
    cat_plot = sns.catplot(x="variable", y="size", hue="value", col="cardio", kind="bar", data=df_cat)

    # 8: Set the correct y-label to 'total'
    cat_plot.set_axis_labels("variable", "total")

    # 9: Save and return the figure
    fig = cat_plot.fig
    fig.savefig('catplot.png')
    return fig

# 10: Draw heatmap
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12: Drop 'BMI' column
    df_heat = df_heat.drop(columns=['BMI'])

    # 13: Calculate the correlation matrix
    corr = df_heat.corr()

    # 14: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 15: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 16: Draw the heatmap and save the figure
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, vmin=-0.1, vmax=0.3, square=True, cbar_kws={'shrink': 0.5})
    fig.savefig('heatmap.png')
    return fig
