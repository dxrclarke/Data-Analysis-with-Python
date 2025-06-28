import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('/medical_examination.csv')

# 2
BMI = df['weight'] / ((df['height'] / 100) ** (2))
df['overweight'] = BMI.apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    columns = [
        'active',
        'alco',
        'cholesterol',
        'gluc',
        'overweight',
        'smoke'
    ]

    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=columns)
    
    #6
    df_cat = df_cat.reset_index() \
                    .groupby(['variable', 'cardio', 'value']) \
                    .agg('count') \
                    .rename(columns={'index': 'total'}) \
                    .reset_index()

    # 7



    # 8
    fig = sns.catplot(x = 'variable', y='total', hue='value', kind='bar', col='cardio', data=df_cat).fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi'])
        & (df['height'] >= df['height'].quantile(0.025))
        & (df['height'] <= df['height'].quantile(0.975))
        & (df['height'] >= df['height'].quantile(0.025))
        & (df['height'] <= df['height'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr(method="pearson")

    # Generate a mask for the upper triangle
    mask = np.triu(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12,12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, linewidths=1, annot = True, square = True, mask = mask, fmt = ".1f", center = 0.08,
                cbar_kws = {"shrink":0.5})
    
    fig.savefig('heatmap.png')
    return fig
