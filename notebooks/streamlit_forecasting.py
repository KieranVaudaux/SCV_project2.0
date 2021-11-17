import streamlit as st
import numpy as np
import scipy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
#import statistics as st
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.distributions.empirical_distribution import ECDF
from TimeSerie_fct import create_monthly_avg_time_serie
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, pacf
from sklearn.utils import resample
import os
#from scipy.stats import bootstrap

import plotly.graph_objects as go

plt.style.use("ggplot")




def st_forecasting():
    
    st.markdown("With this feature, you may run a forecasting model on yearly mean temperatures by regulating the parameters in the sidebar.")
    
    
    
    data_temperature = pd.read_table('../data/observatoire-geneve/TG_STAID000241.txt',sep = ',',
                                names = ['SOUID','DATE','TG','Q_TG'], skiprows = range(0,20))

    data_temperature.drop(data_temperature[ data_temperature['Q_TG'] == 9 ].index, inplace = True)
    data_temperature['Year'] = [int(str(d)[:4]) for d in data_temperature.DATE]
    data_temperature['Month'] = [int(str(d)[4:6]) for d in data_temperature.DATE]
    data_temperature['Day'] = [int(str(d)[6:8]) for d in data_temperature.DATE]

    #Compute the day of the year for each year
    day_of_year = np.array(len(data_temperature['Day']))

    adate = [datetime.strptime(str(date),"%Y%m%d") for date in data_temperature.DATE]
    data_temperature['Day_of_year'] = [d.timetuple().tm_yday for d in adate]
    data_temperature.TG = data_temperature.TG/10.
    
    df = data_temperature.copy()
    
    # Transformation en moyenne annuelle
    Years = df.Year.unique()
    data_Y = pd.DataFrame(np.array([[df[df.Year == y].TG.mean(),
                            df[df.Year == y].TG.median(),df[df.Year == y].TG.std(),y] for y in Years]),
                         index = (np.arange(np.shape(Years)[0])), columns=["Mean","Median","Std","Years"])
    
    
    
    
    break_year = 1962


    mean1 = data_Y.Mean[data_Y.Years<break_year].mean()
    data1 = data_Y[data_Y.Years<break_year].copy()

    mean2 = data_Y.Mean[data_Y.Years>=break_year].mean()
    data2 = data_Y[data_Y.Years>=break_year].copy()

    n = np.shape(data_Y.Mean)[0]
    t = np.array(data_Y.Years).reshape((n,1))
    one = np.ones(shape=(np.shape(t)[0],1))
    X = np.concatenate([one,t],axis = 1)

    n1 = np.shape(data1.Mean)[0]
    n2 = np.shape(data2.Mean)[0]
    t1 = np.array(data1.Years).reshape((n1,1))
    t2 = np.array(data2.Years).reshape((n2,1))
    one1 = np.ones(shape=(np.shape(t1)[0],1))
    X1 = np.concatenate([one1,t1],axis = 1)
    one2 = np.ones(shape=(np.shape(t2)[0],1))
    X2 = np.concatenate([one2,t2],axis = 1)

    acf1 = acf(data1.Mean,nlags=np.shape(data1.Mean)[0],fft=False)
    sigma1 = data1.Mean.var()*sc.linalg.toeplitz(acf1)
    GLS_reg1 = sm.GLS(data1.Mean,X1,sigma1).fit()

    acf2 = acf(data2.Mean,nlags=np.shape(data2.Mean)[0],fft=False)
    sigma2 = data2.Mean.var()*sc.linalg.toeplitz(acf2)
    GLS_reg2 = sm.GLS(data2.Mean,X2,sigma2).fit()

    acf_ = acf(data_Y.Mean,nlags=np.shape(data_Y.Mean)[0],fft=False)
    sigma_ = data_Y.Mean.var()*sc.linalg.toeplitz(acf_)
    GLS_reg = sm.GLS(data_Y.Mean,X,sigma_).fit()

    coef1 = GLS_reg1.params
    coef2 = GLS_reg2.params
    coef = GLS_reg.params

    f = lambda x: coef[0]+coef[1]*x
    f1 = lambda x: coef1[0]+coef1[1]*x
    f2 = lambda x: coef2[0]+coef2[1]*x
    f_split = lambda x: (x<break_year)*f1(x)+(x>=break_year)*f2(x)
    
    
    #First estimate of the covariance matrix using LOWESS to detrend the time series and make it stationary

    n = np.shape(data_Y.Mean)[0]
    frac = np.arange(1./n,1,1./n)
    i = 1

    dens = sm.nonparametric.lowess(data_Y.Mean,np.arange(np.shape(data_Y.Mean)[0]),frac=frac[-i] )
    test_addfuller = adfuller(dens[:,1], maxlag=None, regression='n', autolag='AIC'
                              , store=False, regresults=True)

    while (test_addfuller[1]>0.05):
        i+=1


        dens = sm.nonparametric.lowess(data_Y.Mean,np.arange(np.shape(data_Y.Mean)[0]),frac=frac[-i])
        test_addfuller = adfuller(data_Y.Mean-dens[:,1], maxlag=None, regression='n', autolag='AIC'
                              , store=False, regresults=True)

    #We can reject the fact that the time series is not stationary, therefore we could use the acf
    res = data_Y.Mean-dens[:,1]
    acf_estimate = acf(res,nlags=n,fft=False)
    sigma_estimate = res.var()*sc.linalg.toeplitz(acf_estimate)
    
    
    
    # Regression with adding of a constant in 1962
    tol_conv = 1e-010
    n = np.shape(data_Y.Mean)[0]
    t = np.array(data_Y.Years).reshape((n,1))
    one_1962 = np.array([int(y>=break_year) for y in Years]).reshape((n,1))
    Xx = np.concatenate([one,one_1962,t],axis = 1)


    sigma_x = sigma_estimate  #data_Y.Mean not stationary so we cannot use acf on it 
    sigma_prec = np.eye(n)
    GLS_reg_x = sm.GLS(data_Y.Mean,Xx,sigma_x).fit()
    res = GLS_reg_x.resid

    #We test the stationarity for use the acf
    test_addfuller = adfuller(res, maxlag=None, regression='n', autolag='AIC'
                              , store=False, regresults=True)
    iter_stationarity_bool_x = np.array([bool(test_addfuller[1]<0.05)])

    while (np.linalg.norm(sigma_x-sigma_prec)>tol_conv):
        test_addfuller = adfuller(res, maxlag=None, regression='n', autolag='AIC'
                              , store=False, regresults=True)
        iter_stationarity_bool_x = np.concatenate([iter_stationarity_bool_x,np.array([bool(test_addfuller[1]<0.05)])])
        acf_x = acf(res,nlags=n,fft=False)
        sigma_prec = sigma_x
        sigma_x = res.var()*sc.linalg.toeplitz(acf_x)
        GLS_reg_x = sm.GLS(data_Y.Mean,Xx,sigma_x).fit()

    coef_cst = GLS_reg_x.params
    Log_like_with_cst = GLS_reg_x.llf
    CI_trend = GLS_reg_x.conf_int().loc["x2"]
    #GLS_reg_x.summary()

#The drop on mean temperature in 1962 is of -1.5218 with conf_inf < -1
    
    f_cst = lambda x: coef_cst[0]+coef_cst[1]*(x>=break_year)+coef_cst[2]*x
    
    data_Y["Mean_detrended"] = data_Y.Mean-f_cst(Years)
    
    p = 2
    q = 0

    arma_mod = ARIMA(data_Y.Mean_detrended, order=(p,0,q)).fit(method='innovations_mle')
    res = arma_mod.resid 
    data_Y["resid"] = res
    param = np.array(arma_mod.params)
    data_Y.resid.to_csv("DataGenerated/Annual_resid_ARMA2_0.csv",index=True)
    
    def forecast(data,n_forecast, phi,last):
        
        y = np.arange(int(last)+1,int(last)+n_forecast+1)
        #residus = sc.stats.norm(loc=data.resid.mean(),scale=np.sqrt(phi[-1])).rvs(size=n_forecast)
        residus = np.array(resample(res,n_samples=n_forecast)).reshape((n_forecast,))
        predict = np.array(phi[0]+np.array(data[data.Years == float(last-1)].resid)*phi[2]+
                           np.array(data[data.Years == float(last)].resid)*phi[1] + residus[0])
        predict = np.concatenate([predict,np.array(phi[0]+np.array(data[data.Years == last].resid*phi[2]+
                                                                   predict[0])*phi[1]+ residus[1])])
        for i in range(2,n_forecast):
            predict = np.concatenate([predict,np.array([phi[0]+predict[i-2]*phi[2]+predict[i-1]*phi[1]+ residus[i]])])
        predict = predict + f_cst(y)
        return predict
    
    
    
    nb_simulation = st.sidebar.slider("Number of simulations to run", 1, 300, value=100)
    delta_simulation = st.sidebar.slider("Number of years to predict for", 0, 100, value=10)
    last_year = st.sidebar.slider("Starting point of predictions", 1903, 2020, value=2001) - 1
    
    mean_simulation = np.zeros(shape=(nb_simulation,delta_simulation))
    
    # Simulation of the different forecast
    for i in range(nb_simulation):
        mean_simulation[i,:] = forecast(data_Y, delta_simulation, param, last_year)
    # Confidence interval computed

    CI_forecast = np.zeros(shape=(delta_simulation,2))
    CI_mean = np.zeros(shape=(delta_simulation,2))
    CI_std = np.zeros(shape=(delta_simulation,2))

    for i in range(delta_simulation):
        CI_mean[i:,] = sc.stats.bootstrap((mean_simulation[:,i],),np.mean, confidence_level=0.95).confidence_interval
        CI_std[i:,] = sc.stats.bootstrap((mean_simulation[:,i],),np.std, confidence_level=0.95).confidence_interval
        CI_forecast[i,:] = sc.stats.norm.interval(0.95,loc=CI_mean[i,:].mean(),scale=CI_std[i,1])

    predictions = forecast(data_Y, delta_simulation, param, last_year)
    delta_years = list(range(last_year+1, last_year+1+delta_simulation))
    
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(last_year+1,last_year+delta_simulation+1), y=CI_forecast[:,0],
        fill=None,
        mode='lines',
        name = 'sup CI',
        line_color='rgb(184, 247, 212)',
        ))

    fig.add_trace(go.Scatter(
        x=np.arange(last_year+1,last_year+delta_simulation+1),
        y=CI_forecast[:,1],
        fill='tonexty', # fill area between trace0 and trace1
        name = 'inf CI',
        mode='lines', line_color='rgb(184, 247, 212)'))

    fig.add_trace(go.Scatter(
            x=data_Y.Years[data_Y.Years<=last_year], y=data_Y.Mean[data_Y.Years<=last_year],
            mode='lines',
            name = 'data',
            line=dict(width=1, color='rgb(131, 90, 241)')))

    fig.add_trace(go.Scatter(
            x=[last_year,last_year+1], y=[int(data_Y.Mean[data_Y.Years==last_year]), predictions[0]],
            mode='lines',
            name = 'liaison',
            line=dict(width=1, color='limegreen')))

    fig.add_trace(go.Scatter(
            x=delta_years, y=predictions,
            mode='lines',
            name = 'prediction',
            line=dict(width=1, color='cornflowerblue')))
    
    fig['layout'].update({
        'showlegend': True,
        'width': 950,
        'height': 500,
    })
    fig.update_layout(
    title="FORECASTING RESULT",
    xaxis_title="year",
    yaxis_title="average temperature"
    )
    
    st.plotly_chart(fig)