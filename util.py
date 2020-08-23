darkgrey = '#3A3A3A'
lightgrey = '#414141'
barblue = ''

def prep_libs():
    global barblue
    import pandas as pd
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    
    # %matplotlib inline

    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 1.5

    barblue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    #plt.rcParams['text.color'] = darkgrey
    #plt.rcParams['axes.labelcolor'] = darkgrey
    #plt.rcParams['xtick.color'] = lightgrey
    #plt.rcParams['ytick.color'] = lightgrey