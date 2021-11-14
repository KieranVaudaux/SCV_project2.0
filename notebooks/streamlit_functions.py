from ipywidgets import fixed, interact, interact_manual, interactive
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import base64
import numpy as np
from datetime import datetime
import pandas as pd
from visual_features import *


def display_date_slider(years):
    
    year = st.sidebar.slider('Year', int(min(years)), int(max(years)-1), 1960)
    return year


def st_display_pdf(pdf_file):
    
    with open(pdf_file,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="1000" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    
def main():
    
    st.title("The Whole Story")
    st.subheader("This is the project report, containing all the details of our data analysis.")
    st.write("You have the possibility to download it - to do this, please check the sidebar.")
    
    with open("../reports/SCV_report.pdf","rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    
    
    
def plot_stats_window_st(df,elt):
    
    df['Year'] = [int(str(d)[:4]) for d in df.DATE]
    df['Month'] = [int(str(d)[4:6]) for d in df.DATE]
    df['Day'] = [int(str(d)[6:8]) for d in df.DATE]

    #Compute the day of the year for each year
    day_of_year = np.array(len(df['Day']))

    adate = [datetime.strptime(str(date),"%Y%m%d") for date in df['DATE']]
    df['Day_of_year'] = [d.timetuple().tm_yday for d in adate]

    years = df.Year.unique()
    
    d = [df[df.Year==year]['TG'].mean() for year in years]
    d.pop()
    df_av = pd.DataFrame(d, columns=['ATG'])
    years_ = np.delete(years, years.shape[0]-1)
    df_av['Year'] = years_
    
    years = np.delete(years, years.shape[0]-1)
    year = display_date_slider(years)

    bins = [5*i for i in range(10,21)]
    bin_ = st.sidebar.slider('Bins in histogram', 50, 100, 75)
    
    left_column, right_column = st.columns(2)

    # 1. Mean temperature curve over one year

    fig1, ax1 = plt.subplots(1)
    plot_mean_temp(year=year, ax=ax1, df=df, element=elt)
    left_column.pyplot(fig1, figsize=(10, 10))

    fig2, ax2 = plt.subplots(1)
    plot_hist_mean(year=year, ax=ax2, df=df, element=elt, bins=bin_)
    right_column.pyplot(fig2, figsize=(10, 10))

    fig3, ax3 = plt.subplots(1)
    plot_min(years, x=list(years).index(year), ax=ax3, df=df, element=elt)
    left_column.pyplot(fig3, figsize=(10, 10))

    fig4, ax4 = plt.subplots(1)
    plot_max(years, x=list(years).index(year), ax=ax4, df=df, element=elt)
    right_column.pyplot(fig4, figsize=(10, 10))

    fig5, ax5 = plt.subplots(1)
    plot_std(years, x=list(years).index(year), ax=ax5, df=df)
    left_column.pyplot(fig5, figsize=(10, 10))

    fig6, ax6 = plt.subplots(1)
    pie_chart_missing(years, ax=ax6, df=df)
    st.sidebar.pyplot(fig6,ax6)
    
    fig7, ax7 = plt.subplots(1)
    plot_mean_temp_global(years, x=list(years).index(year), ax=ax7, df_av=df_av, element=elt)
    right_column.pyplot(fig7, figsize=(10, 10))
    
    
    
def description():
    
    st.title('Statistical and Visual Analysis of Meteorological Data')

    st.header('Introduction')

    st.markdown('Hey there! This app is developped in the context of a project supervised by Prof. [Mehdi Gholam](https://people.epfl.ch/mehdi.gholam?lang=fr) at EPFL, in the context of the course *Statistical Computation and Visualization*.')

    st.markdown('Statistical and visual analysis of data are major components of the general data science domain. In this project, we look at various datasets revolving around meteorological recordings from various stations within Switzerland. Statistical analysis of meteorological data plays an important role in understanding and modeling key features in climate change, as well as making short-term predictions on certain meteorological elements. Here, we are interested in providing an efficient pipeline aiming at analysing meteorological data through basic statistical methods such as linear regression and time series analysis. Moreover, we concentrate in providing a significant amount of visualization tools to combine with the statistical results.')

    st.header('Description')
    st.markdown('We are interested in the study of meteorological data from the Geneva Observatory in Switzerland. More specifically, we are interested in the temporal evolution of the average temperature from 1901 to now. We aim to model the evolution of the mean temperature, in order to see if we can observe a significant increasing trend in it. In particular, we use various Python visualisation tools to allow an intuitive interactive framework. The dataset that we use for data analysis can be found [here](https://www.ecad.eu/utils/showselection.php?99j9a2jpggb49ha5t4mc9evpol).')

    st.header('Our Goal')
    st.markdown('Within the statistical data analysis we make, we aim at answering a specific question :')
        
    st.write("_Is there a significant increase in the average temperature trend in Switzerland from 1901 to the present day ?_")


    st.markdown('To try to answer this question, we will first focus on the evolution of the average temperature in Geneva. This will allow us to refine and improve our statistical study on the Geneva observatory data, before extending it to the rest of the weather stations.')
    

def github():
    
    st.sidebar.markdown("The entire code of the project, from source code to notebooks, is available at our GitHub repo [here](https://github.com/LucaNyckees/SCV_project1). Have a look!")
    
def contacts():
    
    st.sidebar.markdown("""
        * Luca Bracone ([EPFL](https://people.epfl.ch/luca.bracone), [GitHub](https://github.com/jkasalt))\n
        * Luca Nyckees ([EPFL](https://people.epfl.ch/luca.nyckees), [GitHub](https://github.com/LucaNyckees))\n
        * Blerton Rashiti [EPFL](https://people.epfl.ch/blerton.rashiti), [GitHub](https://github.com/BlertonRashiti))\n
        * Kieran Vaudaux [EPFL](https://people.epfl.ch/kieran.vaudaux), [GitHub](https://github.com/KieranVaudaux)) 
        """)
    
    
    
def results_display():
    
    st.header("Time Series Analysis - Main Results")
    st.sidebar.header("Time Series Statistical Analysis")

    
    options = ('p-values of the Ljung Box, Box-Pierce test, and McLeod-Li', 
                                         'ACF and PACF',
                                         'QQ-plot',
                                         'p-values of the Mann-Whitney U test')
    
    images = ("p_values_ljung.png",
              "ACF_and_PACF.png",
              "QQ_plot.png",
              "mann_whitney.png")
    
    index = st.selectbox('Which result do you want to see?', range(len(options)), format_func=lambda x: options[x])

    
    elt = options[index]

    st.subheader(elt)
    st.image(images[index])
    descriptions(index)
    

def descriptions(index):
    
    descriptions = ["Such a nice description!",
                   "Such a nice description!",
                   "Such a nice description!",
                   "Such a nice description!"]
    
    with st.expander("See explanation"):
        
        st.markdown(descriptions[index])
   
##  ANNUAL ANALYSIS PART ##################################################

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
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px



def annual_analysis():
    
    data_Y = pd.read_csv("DataGenerated/Annual/Annual_Mean.csv")

    st.title("Evolution of the Mean Temperature at Geneva Observatory")
    
    st.markdown(r'In this section, we will focus on the modelling of our data, which we will see as Times Series.\\We have the daily average temperatures at the Geneva observatory from 1$$^{\text{er}}$$ January 1901 to August 2 2021. That is to say 44044 average temperature record, spread over a period of more than 120 years. As this amount of data is very large, we have chosen to proceed in stages. To do this, we will first look for the presence of a significant increase in the trend in the Time Series of annual mean temperatures at the Geneva observatory, which we have calculated from the daily data. We can then make our study more complex by looking at the time series of monthly, weekly and daily mean temperatures.')
    
    
    st.header("Annual Mean Temperature at Geneva Observatory")
    
    st.markdown('It seems natural to ask whether transforming our data by averaging the annual temperature is relevant. Indeed, knowing that during a year the temperature can vary from $-10\degree$C in winter to more than $30\degree$C in summer, does it really make sense to consider the average of these values? How do we correctly interpret these values and what would it really mean if there was a significant increase in the trend from 1901 to the present? While we have more refined data than annual average temperatures, looking at this one could be debated. However, as many studies also look at annual mean temperatures, we will accept, for the purposes of this project, that the presence of a significant increase in the trend of annual mean temperatures in Geneva would be an additional indication of the presence of climate warming (in Geneva). To confirm this idea, Figure \ref{mean_median_std} below allows us to see that the global behaviour of the Time Series of annual averages is similar to that of the Time Series of annual median temperatures, as well as the standard deviation of the annual average temperatures seems to be homoskedastic. This supports the idea that the annual mean temperatures are not overly affected by the presence of extreme temperatures or by the increase in temperature variability during a year.')
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[1./3,1./3,1./3],
        column_widths=[1],
        specs=[[{"type": "scatter"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}]],
        
        subplot_titles = ['Annual mean temperature',
                        'Annual median temperature',
                        'Annual standard deviation from the mean temperature'],
        x_title = 'Years', y_title = 'Temperature (°C)'
        )

    fig.add_trace(
        go.Scatter(x=data_Y.Years, y=data_Y.Mean, showlegend = False),
            row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x=data_Y.Years, y=data_Y.Median, showlegend = False),
            row=2, col=1
        )
    fig.add_trace(
        go.Scatter(x=data_Y.Years, y=data_Y.Std, showlegend = False),
            row=3, col=1
        )
    
    st.plotly_chart(fig)
    
    st.markdown('The Time Series of average temperatures appears to have an increasing trend over the years, but with a sudden cooling from 1962.\\In order to distinguish more clearly the periods that correspond to a warming or not, we present on Figure \ref{anom_annuelle} the histogram of the annual anomalies that we have standardised. We recall that the temperature anomaly is the difference between the temperature measured in a place (here Geneva), compared to the normal average temperature observed in this same place. ')
    
    anomalie = pd.read_csv("DataGenerated/Annual/Annual_anomalie.csv")
    anom1 = pd.read_csv("DataGenerated/Annual/Annual_anom1.csv")
    anom2 = pd.read_csv("DataGenerated/Annual/Annual_anom2.csv")
    
    
    bar1 = pd.DataFrame()
    bar1["Years"] = data_Y.Years
    bar1["anomalie"] = anomalie.Mean
    bar1["split_anom"] = np.concatenate([np.array(anom1.Mean),np.array(anom2.Mean)])
    xx = np.array(data_Y.Years)
    yy = np.array(anomalie.Mean)
    fig1 = make_subplots(
        rows=3, cols=1,
        row_heights=[1./3,1./3,1./3],
        column_widths=[1],
        specs=[[{"type": "Bar"}],
                [{"type": "Bar"}],
                [{"type": "Bar"}]],
        shared_xaxes = False,
        subplot_titles = ['Annual mean temperature',
                        'Annual median temperature',
                        'Annual standard deviation from the mean temperature'],
        x_title = 'Years', y_title = 'Anomalies'
        )

    fig1.add_trace(
        go.Bar(x=xx, y=yy),
            row=1, col=1
        )
    fig1.add_trace(
        go.Bar(x=xx, y=yy),
            row=2, col=1
        )
    fig1.add_trace(
        go.Bar(x=xx, y=yy),
            row=3, col=1
        )
    st.plotly_chart(fig1)
    
    #bar1 = pd.DataFrame()
    #bar1["Years"] = data_Y.Years
    #bar1["anomalie"] = anomalie
    #bar1["split_anom"] = np.concatenate([np.array(anom1),np.array(anom2)])
    
    #fig = px.bar(bar1,y="anomalie",x="Years")
    #st.plotly_chart(fig)
    
    #st.bar_chart(anomalie[:,1])
#########################################################################################


    
    
