import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
BMI = (df['weight']/(df['height']**2) * 10000) > 25
df['overweight'] = BMI.astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)
df['active'] = (df['active'] == 0).astype(int)



# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, 
                    id_vars=['cardio'],
                    value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                    var_name='variable',
                    value_name='value')


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    cat_plot = sns.catplot(
                        data=df_cat,
                        x='variable',
                        y='total',
                        hue='value',
                        col='cardio',
                        kind='bar'
                    )



    # 8
    fig = cat_plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():

    df_heat = df

    # 11
    df_heat =df_heat[
                        (df_heat['ap_lo'] <= df_heat['ap_hi']) &
                        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
                        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
                        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
                        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
                    ]

    # 12
    corr = df_heat.corr()

    # 13
    # create a triangular matix to mask the correlation matrix for the heat map
    mask = np.triu(np.ones(corr.shape, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize = (12,10))

    # 15

    sns.heatmap(corr, mask = mask, annot=True,       # show correlation values
    fmt=".1f",        # format to 1 decimal place
    cmap='coolwarm',  # color map
    center=0,         # center colormap at 0
    square=True,      # square cells
    linewidths=0.5,   # lines between cells
    cbar_kws={"shrink": 0.5})


    # 16
    fig.savefig('heatmap.png')
    return fig
