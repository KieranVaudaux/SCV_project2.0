import scipy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_monthly_avg_time_serie(data, mean = True):
    new_data = pd.DataFrame()
    years = data.Year.unique()
    months = data.Month.unique()
    for i in range(np.shape(years)[0]):
        for m in range(np.shape(months)[0]):
            avg = data.TG[(data.Year == years[i]) & (data.Month == months[m])].mean()
            new_data = new_data.append([[avg,years[i],months[m]]])
            
    new_data.columns = ["avg_TG","Year","Month"]
    
    return new_data.dropna(axis = 0).copy()
