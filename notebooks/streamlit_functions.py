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
        rows=3, cols=3,
        row_heights=[1./3,1./3,1./3],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "scatter"},None],
                [None,{"type": "scatter"},None],
                [None,{"type": "scatter"},None]],
        subplot_titles = ['Annual mean temperature',
                        'Annual median temperature',
                        'Annual standard deviation from the mean temperature'],
                        x_title = 'Years'
        )

    fig.add_trace(
        go.Scatter(x=data_Y.Years, y=data_Y.Mean, showlegend = False),
            row=1, col=2
        )
    fig.add_trace(
        go.Scatter(x=data_Y.Years, y=data_Y.Median, showlegend = False),
            row=2, col=2
        )
    fig.add_trace(
        go.Scatter(x=data_Y.Years, y=data_Y.Std, showlegend = False),
            row=3, col=2
        )
    fig['layout'].update({
        'width': 1200,
        'height': 600,
    })
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
   
    st.plotly_chart(fig)
    
    st.markdown('The Time Series of average temperatures appears to have an increasing trend over the years, but with a sudden cooling from 1962.\\In order to distinguish more clearly the periods that correspond to a warming or not, we present on Figure \ref{anom_annuelle} the histogram of the annual anomalies that we have standardised. We recall that the temperature anomaly is the difference between the temperature measured in a place (here Geneva), compared to the normal average temperature observed in this same place. ')
    
    anomalie = pd.read_csv("DataGenerated/Annual/Annual_anomalie.csv")
    anom1 = pd.read_csv("DataGenerated/Annual/Annual_anom1.csv")
    anom2 = pd.read_csv("DataGenerated/Annual/Annual_anom2.csv")
    
    fig1 = make_subplots(
        rows=3, cols=3,
        row_heights=[1./3,1./3,1./3],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Bar"},None],
                [None,{"type": "Bar"},None],
                [None,{"type": "Bar"},None]],
        subplot_titles = ['Annual standardized temperature anomalies over the period 1901-2021',
                        'Annual standardized temperature anomalies over the period 1901-1961',
                        'Annual standardized temperature anomalies over the period 1962-2021'],
        x_title = 'Years'
        )

    fig1.add_trace(
        go.Bar(x=data_Y.Years, y=anomalie.Mean,showlegend=False),
            row=1, col=2
        )
    fig1.add_trace(
        go.Bar(x=data_Y[data_Y.Years<1962].Years, y=anom1.Mean,showlegend=False),
            row=2, col=2
        )
    fig1.add_trace(
        go.Bar(x=data_Y[data_Y.Years>=1962].Years, y=anom2.Mean,showlegend=False),
            row=3, col=2
        )
    fig1.update_layout(width=1200, height=600)
    fig1.update_yaxes(title_text="Standardized  anomalies", row=2, col=2)
    st.plotly_chart(fig1)
    
    st.markdown('The visual analysis of these histograms allows us to distinguish four periods 1901-1942, 1943-1961, 1962-1987 and 1988-2021. During the first period, the anomalies tend to be negative, then positive during the second. From the third period, we again observe a cycle of negative and then positive anomalies, but this time more pronounced. This seems to be consistent with the Time Series of average temperatures, in which we had observed an increasing trend but a significant decrease in temperature at the beginning of this third period.')
    
    st.markdown('In order to model our data, we will try to follow the principles of parsimony as much as possible, in order to choose the simplest model that effectively explains our data.')
    
    st.markdown(r'If we denote the Time Series of annual averages by $\{\mathbf{A}_{t}\}_{t}$ $t = 1901,... .2021$, one of the simplest models we could propose is that our observations $\{\mathbf{A}_{t}\}_{t}$ are from a normal distribution, $\mathbf{A}_{t} \stackrel{iid}{\sim} \mathcal{N}(\mu,\sigma^{2})$. To test this we will first compare the empirical distribution of our data with the distribution of a normal distribution of mean $\mathbf{\bar{A}}$ and variance $S^{2}$.')
    
    st.markdown(r'In Figures \ref{qqplot_annuelle} and \ref{ecdf_vs_cdf}, we see that our empirical distribution is quite close to that of a normal distribution, despite the fact that we only have 121 observations.')

    data_Y["ecdf"] = data_Y.Mean
    fig = px.ecdf(data_Y.ecdf)
    
    mean = data_Y.Mean.mean()
    std = data_Y.Mean.std()
    x = np.linspace(min(data_Y.Mean),max(data_Y.Mean),5000)
    
    fig.add_trace(go.Scatter(x = x,
        y =sc.stats.norm.cdf(x,loc = data_Y.Mean.mean(),scale = std),name ='cdf'))
    
    fig['layout'].update({
        'title': 'ECDF vs Normal CDF',
        'title_x': 0.5,
        'xaxis': {
            'title': 'x',
            'zeroline': False
        },
        'yaxis': {
            'title':"P (X<=x)"
        },
        
        'width': 1200,
        'height': 500,
        'legend':{
            'title':""
                         }
    })
    
    st.plotly_chart(fig)
        
    qqplot_data = qqplot(data_Y.Mean, line='s').gca().lines
    ecdf = ECDF(data_Y.Mean)
    fig =  make_subplots(
        rows=1, cols=2,
        row_heights=[1],
        column_widths=[1./2,1./2],
        specs=[[{"type": "Scatter"},
                {"type": "Scatter"}]],
        shared_xaxes = False,
        shared_yaxes = False,
        subplot_titles = ['QQ-Plot of the mean temperature',
                        'Absolute deviation of the ecdf from the cdf']
        )
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    },row = 1, col = 1)

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    },row = 1, col = 1)
    
    fig.add_trace(go.Scatter(x = x,
        y = np.abs(ecdf(x)-sc.stats.norm.cdf(x,loc = data_Y.Mean.mean() ,scale = std)),
        line= dict(color='darkcyan')),
        row = 1, col = 2)
        
    fig.update_xaxes(title_text="Theoritical Quantities", row=1, col=1)
    fig.update_yaxes(title_text="Sample Quantities", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="Absolute deviation", row=1, col=2)
    
    fig['layout'].update({
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'showlegend': False,
        'width': 1200,
        'height': 600,
    })
    
    st.plotly_chart(fig)
    
    
    st.markdown(r'Nevertheless, Figure \ref{acf_pacf} highlights a significant correlation between the annual average temperatures. This leads us to question the independence of the $\mathbf{A}_{t}$ observations. Indeed, if $\{\mathbf{A}_{t}\}_{t=1901}^{2021}$ were independent and identically distributed, we should have that the auto-correlations as well as the partial auto-correlations on Figure \ref{acf_pacf} are approximately in the red zone, which corresponds to an approximate confidence interval for the auto-correlations in the case of an iid sequence.')
    
    st.image("/Users/kieranvaudaux/Documents/SCV/SCV_project2.0/notebooks/figure/Annual_acf_pacf.png")
    
    st.markdown(r'In order to test the independence of our observations, we will use the portmanteau test and several of its variations, namely the Ljung-Box, Box-Pierce and McLeod and Li tests. Figure 1 shows the p-values of these three tests for different numbers in the auto-correlation sequence considered in the test statistics. These tests all have the null hypothesis that the sequence $\{\mathbf{A}_{t}\}_{t=1901}^{2021}$ is iid.')
    
    pval_ind1 = pd.read_csv("DataGenerated/Annual/Annual1_pvalue_indep")
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None]],
        x_title = 'Autocorrelation lags'
        )

    fig.add_trace(go.Scatter(x=pval_ind1.index+1, y=pval_ind1.Ljung,
        mode = 'lines+markers', name = 'Ljung-Box'),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind1.index+1, y=pval_ind1.Pierce,
        mode = 'lines+markers', name = 'Box-Pierce'),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind1.index+1, y=pval_ind1.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li'),row =1, col = 2)
    #fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'P-values of the Ljung-Box ,Box-Pierce test and McLeod-Li',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'yaxis': {
            'type': 'log'
        },
        'width': 1200,
        'height': 500,
    })
    fig.update_yaxes(title_text="p-value", row=1, col=2)
    
    #fig.update_layout(legend=dict(y=0.99,x=0.86))
    
    st.plotly_chart(fig)
    
    st.markdown(r'Since the p-values of these tests are all less than $10^{-11}$, we can conclude that at any significance level greater than $10^{-11}$, the sequence of mean annual temperatures is not from the model $\mathbf{A}_{t} \stackrel{iid}{\sim} \mathcal{N}(\mu,\,\sigma^{2})$.')
    
    st.markdown('To take these dependencies into account, we will consider a more general model from the Time Series study. First of all, we will test the stationarity of our Time Series in order to know which Time Series model could be applied to our data. Without going into formal definitions, a Time Series is stationary if it has a constant mean over time and if its variance is also time invariant.')
    
    st.markdown(r'The test we use to test the stationarity of our Time Series is the Augmented Dickey-Fuller test (ADF), which tests the null hypothesis that a unit root is present in the Time Series, which would make the Time Series non-stationary. This test gives us a p-value of $p = 0.8711$, which is far from significant. This result tends to make us think that the Time Series is not stationary, which can certainly be explained by the presence of an increasing trend that we had already noticed visually on Figure \ref{mean_median_std}. To test this we use another version of the Augmented Dickey-Fuller test to test the trend-stationarity of the Time Series. With this test we obtain a p-value of $p = 0.0162$ which is significant, at the standard significance level of $\alpha = 0.05$ for example.')
    
    st.markdown(r'Following this result, we are therefore led to first model the trend of our Time Series before trying to model our data with a stationary Time Series model. To model our trend we use a generalized linear regression, i.e. a linear regression in which we do not assume the independence of our errors. We chose to model the trend as an affine function of time $\mathbf{A}_{t} = \beta_{0}+\beta_{1}t + \epsilon_{t}$ with $\mathbf{\epsilon} = (\epsilon_{1901},...,\epsilon_{2021})^{T}\sim \mathcal{N}(0,\mathbf{\Sigma})$, so as to keep the model simple and to be able to easily infer the sign of $\beta_{1}$, which will allow us to detect or not a significant growth of the mean temperature trend. To do this, we estimated the covariance matrix of $\mathbf{\epsilon}$ by the Toeplitz matrix generated by the sequence of auto-correlations of our Time Series. Figure \ref{full_GLS}, shows us the fit of the trend estimate by a line. We notice visually that the fit is quite good overall, but that the line has difficulty in approximating the period 1960-1990 correctly. This is due to the fact that, as we saw in Figure \ref{mean_median_std}, the mean annual temperature drops sharply in 1962 before resuming a "normal" behaviour in relation to the rest of the Time Series.')
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None]],
        x_title = 'Years'
        )
    
    trendless_gls = pd.read_csv("DataGenerated/Annual/Annual1_trendlessMean_GLS")
    
    fig.add_trace(go.Scatter(x = data_Y.Years, y = data_Y.Mean,
        name = 'Annual mean temperature'),row = 1, col = 2)
    fig.add_trace(go.Scatter(x = data_Y.Years, y = trendless_gls.f_without_cts,
        name = 'GLS without constant'),row = 1, col = 2)
    fig.add_trace(go.Scatter(x = data_Y.Years, y = trendless_gls.f_cts,
        name = 'GLS with constant',
        marker = dict(color = 'mediumvioletred')),row = 1, col = 2)
    
    fig['layout'].update({
        'title': 'Modelisation of the trend by GLS',
        'title_x': 0.5,
        'xaxis': {
            'title': 'Years',
            'zeroline': False
        },
        'width': 1200,
        'height': 500,
    })
    
    fig.update_yaxes(title_text="Annual mean temperature (°C)", row=1, col=2)
        
    st.plotly_chart(fig)
    
    st.markdown('Initially, we wondered whether this sudden change was due to a change in equipment or to the automation of the temperature recordings, which could have affected the average daily temperatures and, by transitivity, the average annual temperatures. But, given the very large temperature difference, this seems unlikely and we have not found any information that would support this hypothesis. However, by doing some additional research we were able to find articles and websites on which the years 1962-1964 were described as particularly cold years with harsh winters. This sudden cooling could then be simply due to a temporary local cooling. We then considered this sudden cooling as a rare event, and in order not to affect our study too much, we chose to estimate the trend of the annual mean temperatures over the periods 1901-1961 and 1961-2021 separately. This led to the estimate in Figure 2 below.')
    
    st.markdown(r'The results of the two GLS estimates are summarised in the tables ref{split_GLS_summary1} and ref{split_GLS_summary2} which represent the results for the period 1901-1961 and the period 1962-2021 respectively.In the table ref{split_GLS_summary1}, we see that the p-value of the t-test is $p = 0.1$, on the first period we can only reject the null hypothesis, that $\beta_1=0$, at a significance level of $\alpha = 0.1$ . However, the 95% confidence interval which is given contains $0$, so we cannot conclude anything about the positivity of $\beta_1$ at the threshold of $\alpha = 0.05$ on this period.')
    
    
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader('GLS Regression Results without constant in 1962')
        HtmlFile = open("DataGenerated/Annual/Annual_summa_without_cts_personaliser", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.markdown(source_code, unsafe_allow_html=True)
 
    
    st.markdown(r'On the other hand, the table \ref{split_GLS_summary2} allows us to reject the null hypothesis, which is that $\beta_{1} = 0$ over the period 1962-2021, at the significance level $\alpha = 0.05$ because we obtain a p-value associated with the t-test of $\beta_{1}$ of $p = 0.042$. Moreover, the 95\% confidence interval of $\beta_{1}$ is [0.001,0.070], which contains only strictly positive values. If our model subsequently proves to be consistent with our data, then we would have a significant indication of an increasing trend in mean temperatures over the period 1962-2021.')
        
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader('GLS Regression Results with constant in 1962')
        HtmlFile = open("DataGenerated/Annual/Annual_summa_with_cts_personaliser", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.markdown(source_code, unsafe_allow_html=True)
    
    
    st.markdown(r'We will now consider the Time Series $\{\mathbf{A}_{t}\}_{t=1901}^{2021}$ which is the Time Series $\{\mathbf{A}_{t}\}_{t=1901}^{2021}$ from which we have subtracted the trend calculated above, over two disjoint periods. Figure \ref{dentrend_mean} allows us to visualise this new Time Series.')
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./8,3./4,1./8],
        specs=[[None,{"type": "Scatter"},None]],
        shared_xaxes = False,
        shared_yaxes = False,
        )
    fig.add_trace(go.Scatter(x = data_Y.Years, y = trendless_gls.with_cts),row = 1, col =2)
    
    fig['layout'].update({
        'title': 'Trendless mean temperature',
        'title_x': 0.5,
        'xaxis': {
            'title': 'Years',
            'zeroline': False
        },
        'yaxis': {
            'title':'Trendless mean temperature (°C)'
        },
        'width': 1200,
        'height': 500,
    })
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    st.plotly_chart(fig)
    
    st.markdown(r'Therefore, it is interesting to test again the independence of the elements of the new Time Series. For this we proceed as before, we have calculated the auto-correlation and partial auto-correlation sequences of the new Time Series, which can be seen in Figure \ref{dentrend_acf}. Almost all terms of the sequence are within the approximate confidence interval in red, but one can see that some terms are still outside. Moreover, for a number of lags greater than 20, we see in Figure \ref{dentrend_LjungTest} that the p-values associated with the Ljung-Box and Box-Pierce tests allow us to reject the null hypothesis of independence of the Time Series elements. ')
    
    st.image("/Users/kieranvaudaux/Documents/SCV/SCV_project2.0/notebooks/figure/Detrended_mean_acf_pacf_cst.png")
    
    pval_ind2 =pd.read_csv(
    "/Users/kieranvaudaux/Documents/SCV/SCV_project2.0/notebooks/DataGenerated/Annual/Annual2_pvalue_indep_GLS")
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None]],
        x_title = 'Autocorrelation lags'
        )

    fig.add_trace(go.Scatter(x=pval_ind2.index+1, y=pval_ind2.Ljung,
        mode = 'lines+markers', name = 'Ljung-Box',
                    marker=dict(size=8)),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind2.index+1, y=pval_ind2.Pierce,
        mode = 'lines+markers', name = 'Box-Pierce',
                    marker=dict(size=8)),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind2.index+1, y=pval_ind2.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li',
                    marker=dict(size=8)),row =1, col = 2)
    fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'P-values of the Ljung-Box ,Box-Pierce test and McLeod-Li on the trendless mean temperature',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'yaxis': {
            'type': 'log'
        },
        'width': 1200,
        'height': 500,
    })
    fig.update_yaxes(title_text="p-value", row=1, col=2)
    
    #fig.update_layout(legend=dict(y=0.99,x=0.86))
    
    st.plotly_chart(fig)
    
    st.markdown(r'Even though removing the trend from the Time Series of mean annual temperatures has reduced the dependence between observations, there is still enough dependence between observations to use a Times Series model to model $\{\mathbf{\Tilde{A}}_{t}\}_{t=1901}^{2021}$. Especially since this new Time Series is now stationary. Indeed, the Augmented Dickey-Fuller test on this one gives us a p-value of $p = 2.0748e-08$ and thus allows us to reject the null hypothesis of non-stationarity at a significance level of $\alpha = 0.05$.')
    
    st.markdown(r'Figure \ref{detrended_modelSelection} allows us to compare several models for the Time Series $\{\mathbf{\Tilde{A}}_{t}\}_{t=1901}^{2021}$ thanks to the Akaike information criterion (AIC). We have tried to model the Time Series as arising from an $ARMA(p,q)$ for $p\in\{0,...,3\}$ and $q\in\{0,...,9\}$. Our choice of restricting $p$ and $q$ is mainly due to the fact that our Time Series is not very large and that we wanted to try to keep the model as simple as possible to model our data.')
    
    
    aic =pd.read_csv(
    "/Users/kieranvaudaux/Documents/SCV/SCV_project2.0/notebooks/DataGenerated/Annual/Annual_modelSelection.csv")
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None]],
        x_title = 'MA(q) parameters'
        )
    q = np.arange(0,10)
    for i in range(4):
        fig.add_trace(go.Scatter(x=q, y=aic[aic.param_p==i].AIC,
            mode = 'lines+markers', name = 'p = '+str(i),
            marker=dict(size=15)),row =1, col = 2)
    
    #fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'Time series model selection for the trendless mean',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'yaxis': {
            'type': 'log'
        },
        'width': 1200,
        'height': 500,
    })
    fig.update_yaxes(title_text="AIC", row=1, col=2)
    fig.update_layout(hovermode="x unified")
    
    #fig.update_layout(legend=dict(y=0.99,x=0.86))
    
    st.plotly_chart(fig)

    st.markdown(r'By choosing the model which minimises the AIC, we are led to consider the model $ARMA(0,2)$ to model the Time Series $\{\mathbf{\tilde{A}}_{t}\}_{t=1901}^{2021}$, which amounts to considering a model $MA(2)$.')
    
    resid_arma2_0 =pd.read_csv("DataGenerated/Annual_resid_ARMA2_0.csv")
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./8,3./4,1./8],
        specs=[[None,{"type": "Scatter"},None]],
        shared_xaxes = False,
        shared_yaxes = False,
        )
    fig.add_trace(go.Scatter(x = data_Y.Years, y = resid_arma2_0.resid),
        row = 1, col =2)
    
    fig['layout'].update({
        'title': 'Residus of the ARMA(2,0) model',
        'title_x': 0.5,
        'xaxis': {
            'title': 'Years',
            'zeroline': False
        },
        'yaxis': {
            'title':'Residus mean temperature(°C)'
        },
        'width': 1200,
        'height': 500,
    })
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    st.plotly_chart(fig)
    
    st.markdown(r'In order to confirm the consistency of our model, we tested the independence and distribution of the residuals obtained. Figure \ref{Ljungbox_residu} allows us to see that the residuals of our model do not allow us to reject the null hypothesis of independence of the Ljung-Box, Box-Pierce and McLeod-Li tests, at a significance level of $\alpha = 0.05$, except for the McLeod-Li test which rejects the null hypothesis for a value of the auto-correlation lags.')
    
    pval_ind3 =pd.read_csv(
    "/Users/kieranvaudaux/Documents/SCV/SCV_project2.0/notebooks/DataGenerated/Annual/Annual2_pvalue_indep_ARMA2_0")
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None]],
        x_title = 'Autocorrelation lags'
        )

    fig.add_trace(go.Scatter(x=pval_ind3.index+1, y=pval_ind3.Ljung,
        mode = 'lines+markers', name = 'Ljung-Box',
                    marker=dict(size=8)),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind3.index+1, y=pval_ind3.Pierce,
        mode = 'lines+markers', name = 'Box-Pierce',
                    marker=dict(size=8)),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind3.index+1, y=pval_ind3.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li',
                    marker=dict(size=8)),row =1, col = 2)
    fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'P-values of the Ljung-Box ,Box-Pierce test and McLeod-Li on the residus of the ARMA(2,0)',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'width': 1200,
        'height': 500,
    })
    fig.update_yaxes(title_text="p-value", row=1, col=2)
    
    #fig.update_layout(legend=dict(y=0.99,x=0.86))
    
    st.plotly_chart(fig)
    
    
    st.markdown(r'Now that our data appear to be independent we can use the Mann-Whitney U test, which compares the distribution of the first $n$ of data with the distribution of the rest of the data. This test has the null hypothesis that the probability that a variable generated by the first distribution is greater than a variable generated by the second distribution is equal to the probability that a variable generated by the second distribution is greater than a variable generated by the first distribution. In Figure \ref{MannWhitney}, we have the p-values obtained by this test for different values of $n = 1,...,119$.')
    
    st.markdown(r'We therefore seem to have obtained residuals that are independent of each other. Moreover, after performing the Jacques-Bera test, which has the null hypothesis that our data are from a normal distribution, and the Goldfeld-Quandt test, which has the null hypothesis that the data are homoskedastic, we get the p-values : ')
    
    
    
    
#########################################################################################


    
    
