import matplotlib.pyplot as plt
#import pandas as pd


def explore_plot(df):
    df.plot(subplots=True,figsize=(25,10))
    plt.show()