import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('garments_worker_productivity (1).csv')

def scatter_plot(input1,input2):
    x = data[input1]
    y = data[input2]
    plt.scatter(x, y)
    plt.show()

#scatter_plot('day','date')

'''def box_plot(wip, smv, over_time, targeted_productivity, actual_productivity, incentive):
    b = data['wip']
    a = data['smv']
    w = data['over_time']
    x = data["targeted_productivity"]
    y = data["actual_productivity"]
    z = data['incentive']

    columns = [a, b, w, x, y, z]
    fig, ax = plt.subplots()
    ax.boxplot(columns)
    plt.show()'''

'''box_plot('targeted_productivity', 'actual_productivity', 'incentive', 'over_time', 'smv', 'wip')'''

def line_plot(input1, input2):
    s = data[input1]
    t = data[input2]
    plt.plot(s, t)
    plt.title('Line Plot')
    plt.xlabel(input1)
    plt.ylabel(input2)
    plt.show()

#line_plot('day', 'department')

