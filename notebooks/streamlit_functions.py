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
import networkx as nx
from pyvis.network import Network
from stvis import pv_static
from scipy.stats.stats import pearsonr 
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from plotly_features import plotly_mean_temp, plotly_hist_mean, plotly_min, plotly_max, plotly_std, plotly_mean_temp_global

from streamlit_forecasting import st_forecasting
import streamlit.components.v1 as components


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
    
    
    
def correlation_net(df, elt):
    
    st.subheader("Correlation network feature")
    
    st.markdown("We plot the correlation matrix between each year's mean temperature vector.")
    
    df['Year'] = [int(str(d)[:4]) for d in df.DATE]
    df['Month'] = [int(str(d)[4:6]) for d in df.DATE]
    df['Day'] = [int(str(d)[6:8]) for d in df.DATE]
    
    values = st.sidebar.slider('Window of years to correlate',1901, 2020, (1990, 2020))
    
    nb_nodes = len(range(values[0],values[1]+1))
        
    corr = np.zeros((nb_nodes,nb_nodes))

    for i in range(nb_nodes):
        for j in range(nb_nodes):

            year1 = i+values[0]
            year2 = j+values[0]
            if len(df[df.Year==year1]['TG']) == len(df[df.Year==year2]['TG']):
                corr[i,j] = pearsonr(df[df.Year==year1]['TG'],df[df.Year==year2]['TG'])[0]
           
                
    G = nx.Graph()
    nodes = [i+values[0] for i in range(nb_nodes)]
    G.add_nodes_from(nodes)
    
    edges = []
    
    threshold = st.sidebar.slider("Correlation threshold", 0.0, 1.0, value=0.8)

    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if corr[i,j]>threshold and i!=j:
                edges.append((i+values[0],j+values[0]))


    G.add_edges_from(edges)
    

    nt = Network("340px", "860px",notebook=True)
    nt.from_nx(G)
    
    l,m,r = st.columns([2,1,1])
    #data__=np.array([[1, 25, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, 5, 20]])
    fig = px.imshow(corr,
                labels=dict(x="year", y="year", color="Correlation"),
                x=list(range(values[0],values[1]+1)),
                y=list(range(values[0],values[1]+1))
               )
    fig.update_layout(
    title="Correlation matrix - a vector is one year's mean temperatures"
    )
    
    l.plotly_chart(fig)
    
    with st.expander("See explanation"):
        
        st.markdown("For $k\in\{1901,1902,...,2020\}$, let $\mathbf{x}_k$ be the vector of mean temperatures recorded in year $k$. we define the *sample Pearson correlation coefficient between two vectors* $x$ and $y$ of size $n$ as the quantity")
        st.latex(r'''\mathrm{Corr}(\mathbf{x}_i,\mathbf{x}_j)=\frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^n (x_i-\bar{x})^2}\sqrt{\sum_{i=1}^n (y_i-\bar{y})^2}},\\
        \text{where } \bar{v} \text{ denotes the mean } \frac{1}{n}\sum_{i=1}^n v_i \text{ of a vector } v=(v_i)_{i=1}^n.
        ''')
        st.markdown("For two integer indices $i,j\in\{1901,1902,...,2020\}$, we define a matrix entry $M_{i,j}$ as $\mathrm{Corr}(\mathbf{x}_i,\mathbf{x}_j)$. Now, depending on the choice of time-window $\Omega=\{m_{start},m_{start}+1,...,m_{end}\}$ (*i.e.* a finite range of consecutive years), we plot the correlation matrix $M=(M_{i,j})_{i,j\in\Omega}$ as a regular heatmap (*cf.* the colorbar on the side).")
        
    
    st.markdown("For each slider value, we plot a threshold-network induced by the correlation matrix between each year.")
        
        
    with st.expander("Show parametrized correlation network"):
        
        pv_static(nt)
        
    with st.expander("See explanation"):
        
        st.markdown("Based on the correlation matrix $M$ defined above, and a choice of threshold value $C\in[0,1]$, we define a graph $G=G_{(M,C)}$ as follows. For each year $i\in\Omega$ (where $\Omega$ is the time-window as above), we create a node $i$ in $G$. Moreover, for each pair of nodes $n_i,n_j$ in $G$, we draw an edge $(n_i,n_j)$ if and only if we have $M_{i,j}\geq C$, *i.e* if the years $i$ and $j$ are correlated enough.")
        
        
def multiple_curves_window(df,elt):
    
    st.subheader("Time-window visualization")
    
    st.markdown("For visual simplicity and better interpretation, we recommend to choose a small time-window (e.g. 2 to 4 years).")
                
    left_column, right_column = st.columns(2)

    df['Year'] = [int(str(d)[:4]) for d in df.DATE]
    df['Month'] = [int(str(d)[4:6]) for d in df.DATE]
    df['Day'] = [int(str(d)[6:8]) for d in df.DATE]

    #Compute the day of the year for each year
    day_of_year = np.array(len(df['Day']))

    adate = [datetime.strptime(str(date),"%Y%m%d") for date in df['DATE']]
    df['Day_of_year'] = [d.timetuple().tm_yday for d in adate]
    
    fig1 = go.Figure()
    fig2 = go.Figure()
    
    fig1['layout'].update({
        'showlegend': True,
        'width': 600,
        'height': 500,
    })
    fig2['layout'].update({
        'showlegend': True,
        'width': 600,
        'height': 500,
    })
    
    fig1.update_layout(
    title="TIME-WINDOW VIEWPOINT : CURVES",
    xaxis_title="day of the year",
    yaxis_title=elt
    )
    
    fig2.update_layout(
    title="TIME-WINDOW VIEWPOINT : HISTOGRAMS",
    xaxis_title="sunshine duration",
    yaxis_title="count of days"
    )
    
    values = st.sidebar.slider('Select a range of years',1901, 2020, (1965, 1967))
    #bins = [5*i for i in range(10,21)]
    bin_ = st.sidebar.slider('Bins in histogram', 50, 100, 75)

    for year in range(values[0],values[1]+1):
        
        plotly_mean_temp(year=year, fig=fig1, df=df, elt=elt)
        plotly_hist_mean(year=year, fig=fig2, df=df, elt=elt, bins=bin_, iterate=True)
        
    #fig2.update_layout(barmode='stack')
    
    left_column.plotly_chart(fig1)
    right_column.plotly_chart(fig2)
    
    with st.expander("See explanation"):
        
        st.markdown("For a choice of time-window $\Omega=\{m_{start},m_{start}+1,...,m_{end}\}$ (*i.e.* a finite range of consecutive years), we produce two plots. On the left, we show the evolution curve $\{(x,f_T(x)):x\in Y\}\subset\mathbf{R}^2$, for each year $T\in\Omega$, where the function $f_T$ assigns to each day $x$ of year $T$ its recorded mean temperature. On the right, we show the histogram distribution of the data. More precisely, depending on a choice of time-window $\Omega$ and a number of bins $N$, we separate the axis of mean temperature values into $N$ regular-sized intervals and create bins accordingly. For each year $T\in\Omega$, and for each interval $I$, we create and plot a bin $B_I$ whose height represents the counting index $n(I)=|\{x\in T:f_T(x)\in I\}|$.")
    
    
    
    
    
    
def plot_stats_window_st(df,elt):
    
    st.subheader("Interactive descriptive summary")
    
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

    #bins = [5*i for i in range(10,21)]
    bin_ = st.sidebar.slider('Bins in histogram', 50, 100, 75)
    
    left_column, right_column = st.columns(2)

    # 1. Mean temperature curve over one year
    


    fig1 = go.Figure()
    plotly_mean_temp(year=year, fig=fig1, df=df, elt=elt)
    left_column.plotly_chart(fig1)
    
    fig2 = go.Figure()
    plotly_hist_mean(year=year, fig=fig2, df=df, elt=elt, bins=bin_)
    right_column.plotly_chart(fig2)

    fig3 = go.Figure()
    plotly_min(years, x=list(years).index(year), fig=fig3, df=df, elt=elt)
    left_column.plotly_chart(fig3)

    
    fig4 = go.Figure()
    plotly_max(years, x=list(years).index(year), fig=fig4, df=df, elt=elt)
    right_column.plotly_chart(fig4)
    
    fig5 = go.Figure()
    plotly_std(years, x=list(years).index(year), fig=fig5, df=df)
    left_column.plotly_chart(fig5)
    
    x=list(years).index(year)
    fig6 = px.pie(df[df['Year']==years[x]], values='Q_TG')
    #plotly_pie_chart_missing(years, x=list(years).index(year), fig=fig6, df=df)

    labels = 'Recorded', 'Missing'
    df_ = df[df['Year']==years[x]]
    values = [len(df_)-len(df_[df_['Q_TG']==9]), len(df_[df_['Q_TG']==9])]
    fig6 = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig6['layout'].update({
        'showlegend': True,
        'width': 400,
        'height': 400,
    })
    fig6.update_layout(
    title="Proportion of missing values"
    )
    st.sidebar.plotly_chart(fig6)
    
    fig7 = go.Figure()
    plotly_mean_temp_global(years, x=list(years).index(year), fig=fig7, df_av=df_av, element=elt)
    right_column.plotly_chart(fig7)
    
    with st.expander("See interpretation"):
        
        st.markdown("""We observe a two-step increasing pattern for the mean temperature, in the sense that there the average temperature curve oscillates around a line $l_1$ with positive derivative, then around a line $l_2$ with positive derivative too. The transition happens around the year 1963, where we observe a heavy drop in temperatures. This motivates us to fit models in two steps : one for the period 1901-1963 and the other one for the period 1964-1965. The maximum mean temperature presents a similar two-phase pattern, with higher amplitude oscillations during the first phase. Also, note that for a given year, we almost consistently observe the same mean temperature curve. The sunshine duration shows a constant increase from the year 1980 up to now, with a more complex pattern in the years before that.""")

    
    
    
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
    
def datasets():
    
    st.sidebar.markdown("The data we work with can be found [here](https://www.ecad.eu/utils/showselection.php?99j9a2jpggb49ha5t4mc9evpol).")
    

def github():
    
    st.sidebar.markdown("The entire code of the project, from source code to notebooks, is available at our GitHub repo [here](https://github.com/LucaNyckees/SCV_project1). Have a look!")
    
def contacts():
    
    st.sidebar.markdown("""
        * Luca Nyckees ([EPFL](https://people.epfl.ch/luca.nyckees), [GitHub](https://github.com/LucaNyckees))\n
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

import pingouin as pg


def annual_intro():
    st.title("Evolution of the Mean Temperature at Geneva Observatory")
    

    st.markdown(r'In this section, we will focus on the modelling of our data, which we will see as Times Series.\\We have the daily average temperatures at the Geneva observatory from 1$$^{\text{rst}}$$ January 1901 to August 2 2021. That is to say 44044 average temperature record, spread over a period of more than 120 years. As this amount of data is very large, we have chosen to proceed in stages. To do this, we will first look for the presence of a significant increase in the trend in the Time Series of annual mean temperatures at the Geneva observatory, which we have calculated from the daily data. We can then make our study more complex by looking at the time series of monthly, weekly and daily mean temperatures.')


def annual_analysis():
    
    data_Y = pd.read_csv("DataGenerated/Annual/Annual_Mean.csv")
    
    c1,c2,c3 = st.columns([1,10,1])
    with c2:
        st.title("Annual Mean Temperature at Geneva Observatory")
    
    st.markdown('It seems natural to ask whether transforming our data by averaging the annual temperature is relevant. Indeed, knowing that during a year the temperature can vary from $-10\degree$C in winter to more than $30\degree$C in summer, does it really make sense to consider the average of these values? How do we correctly interpret these values and what would it really mean if there was a significant increase in the trend from 1901 to the present day? While we have more refined data than annual average temperatures, looking at this one could be debated. However, as many studies also look at annual mean temperatures, we will accept, for the purposes of this project, that the presence of a significant increase in the trend of annual mean temperatures in Geneva would be an additional indication of the presence of climate warming (in Geneva). To confirm this idea,the following figure allows us to see that the global behaviour of the Time Series of annual averages is similar to that of the Time Series of annual median temperatures, as well as the standard deviation of the annual average temperatures seems to be homoskedastic. This supports the idea that the annual mean temperatures are not overly affected by the presence of extreme temperatures or by the increase in temperature variability during a year.')
    
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
    
    st.markdown(r'''The Time Series of average temperatures appears to have an increasing trend over the years, but with a sudden cooling from 1962.''')
    st.markdown(r'In order to distinguish more clearly the periods that correspond to a warming or not, we present on the following figure the histogram of the annual anomalies that we have standardized.')
    
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
    
    with st.expander("Standardized anomalies explanation"):
        st.markdown(r'The temperature anomaly is the difference between the temperature measured in a place (here the mean temperature at Geneva), compared to the normal average temperature observed in this same place over a certain period (here we look at the periods 1901-2021, 1901-1961, and 1962-2021). On a given period, the standardized anomalies are computed by normalizing the anomalies by the standard deviation of the temperature from the mean temperature of this period.')
    
    st.markdown('The visual analysis of these histograms allows us to distinguish four periods 1901-1942, 1943-1961, 1962-1987 and 1988-2021. During the first period, the anomalies tend to be negative, then positive during the second. From the third period, we again observe a cycle of negative and then positive anomalies, but this time more pronounced. This seems to be consistent with the Time Series of average temperatures, in which we had observed an increasing trend but a significant decrease in temperature at the beginning of this third period.')
    
    st.markdown('In order to model our data, we will try to follow the principles of parsimony (Occam\'s razor) as much as possible, in order to choose the simplest model that effectively explains our data.')
    
    st.markdown(r'If we denote the time series of annual averages by $\{\mathbf{A}_{t}\}_{t}$ for $t = 1901,... .2021$, one of the simplest models we could propose is that our observations $\{\mathbf{A}_{t}\}_{t}$ are from a normal distribution, $\mathbf{A}_{t} \stackrel{iid}{\sim} \mathcal{N}(\mu,\sigma^{2})$. To test this we will first compare the empirical distribution of our data with the distribution of a normal distribution of mean $\mathbf{\bar{A}}$ and variance $S^{2}$.')
    
    st.markdown(r'On the figures below, we see that our empirical distribution is quite close to that of a normal distribution, despite the fact that we only have 121 observations.')

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
    import pingouin as pg
    qqplot_data = qqplot(data_Y.Mean,loc=data_Y.Mean.mean(),scale = data_Y.Mean.std(), line='45').gca().lines
    ecdf = ECDF(data_Y.Mean)
    fig =  make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./4,1./2,1./4],
        specs=[[None,
                {"type": "Scatter"},None]],
        shared_xaxes = False,
        shared_yaxes = False,
        subplot_titles = ['Absolute deviation of the ECDF from the CDF']
        )
    fig.add_trace(go.Scatter(x = x,
        y = np.abs(ecdf(x)-sc.stats.norm.cdf(x,loc = data_Y.Mean.mean() ,scale = std)),
        line= dict(color='darkcyan')),
        row = 1, col = 2)
        
    #fig.update_xaxes(title_text="Theoritical Quantities", row=1, col=1)
    #fig.update_yaxes(title_text="Sample Quantities", row=1, col=1)
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
    c1,c2,c3 = st.columns([1.3,2,1])
    with c2:
        HtmlFile = open("tempQQplot.html", 'r', encoding='utf-8')

        source_code = HtmlFile.read()

        components.html(source_code, height = 500)
    
    st.markdown(r'In order to quantify and test the dependencies between temperatures, our observations must be stationary so that we can calculate the sequences of autocorrelations and partial autocorrelations of our data and make some statistical tests on it. To test the stationarity of our data the use the Augmented Dickey-Fuller test (ADF), which tests the null hypothesis that a unit root is present in a time series sample. Performing the ADF on our data we get a p-value of $p_{value} = 0.871$, which allows us to reject the null hypothesis of stationarity.')
    
    
    st.markdown(r'''Nevertheless, as stated in the book "Time Series Analysis With Applications in R" (p.125) ([[4]](https://link.springer.com/book/10.1007/978-0-387-75959-3?token=M3aff43&utm_campaign=3_fjp8312_springer_katte_M3aff43&countryChanged=true&gclid=Cj0KCQjw5oiMBhDtARIsAJi0qk0A0ZQ5Ip1dgVTIeG-NgmclH396p2BwCJeMZJAwQMk1iW_ct9GaPnEaAiudEALw_wcB)), the sample auto-correlation function (ACF) computed for nonstationary series will also usually indicate the nonstationarity.Indeed, for nonstationary series, the sample ACF typically fails to die out rapidly as the lags increase. This is due to the tendency for nonstationary series to drift slowly, either up or down, with apparent “trend”. With this in mind, the followings figure of the ACF and the partial auto-correlation function (PACF) highlights some "significants" correlations between the annual averages temperatures. This leads us to question the independence of the $\mathbf{A}_{t}$ observations. Indeed, if $\{\mathbf{A}_{t}\}_{t=1901}^{2021}$ were independent and identically distributed, we should have that the ACF and the PACF should lies in the blue zone on the figure below.''')
    
    st.image("figure/Annual_acf_pacf.png")
    
    with st.expander("Explanation of the test for the acf and pacf plot"):
        st.markdown(r'On the two plot above, the blue zone corresponds to an approximate confidence interval for the acf and pacf under the null hypothesis that the acf and pacf are computedf from a iid sequence. This test is based on the fact that for large n the sample autocorrelations of an iid sequence $Y_1, . . . , Y_n$ with finite variance are approximately iid with distribution $\mathbb{N}(0, \frac{1}{n})$. We can therefore test whether or not the observed residuals are consistent with iid noise by examining the sample autocorrelations and partial autocorrelations of the residuals and rejecting the iid noise hypothesis if more than three or four out of 50 fall outside the bounds $\pm \frac{1.96}{\sqrt{n}}$ or if one falls far outside the bounds.')
    
    st.markdown(r'Even if the mean temperature aren\'t stationary, the acf and pacf suggest that we should question the independence of the mean temperature. In order to test the independence of our observations, we will use the portmanteau test and several of its variations, namely the Ljung-Box and McLeod-Li tests, but we should always keep in mind that this tests are also based on the acf of the mean temeprature and thus should be used with caution since our data seem to be nonstationary. The figure below shows the p-values of these two tests for different numbers in the auto-correlation sequence considered in the test statistics. These tests all have the null hypothesis that the sequence $\{\mathbf{A}_{t}\}_{t=1901}^{2021}$ is iid.')
    
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
    fig.add_trace(go.Scatter(x=pval_ind1.index+1, y=pval_ind1.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li'),row =1, col = 2)
    #fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'P-values of the Ljung-Box and McLeod-Li tests',
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
    
    with st.expander("Explanation of the Ljung-Box and McLeod-Li tests"):
        st.markdown(r'''
        The idea behind these two tests is, instead of checking to see whether each sample autocorrelation $\rho (j)$ falls inside the bounds $\pm \frac{1.96}{\sqrt{n}}$, it is also possible to consider the either the statistic $\mathbf{Q}_{LB} = n(n+2)\sum_{j=1}^{h} \frac{\rho^2 (j)}{n-j}$ for the Ljung-Box tests or the statistic $\mathbf{Q}_{ML} = n(n+2)\sum_{j=1}^{h} \frac{\rho_{ww}^2 (j)}{n-j}$ for the McLeod-Li test, where $\rho (j)$ is the acf and $\rho_{ww}(j)$ if the acf the squared data. Then, under the null hypothesis that our data are iid, this both statistics could be approximated by a chi-squared distribution with $h$ degrees of freedom.
        ''')
    
    st.markdown(r'Since the p-values of these tests are all less than $10^{-8}$, we can conclude that we can reject the null hypothesis of idd, at any significance level greater than $10^{-8}$. Thus, even is our data do not seem to be stationary, the behaviour of the acf and pacf do not appear to be in favour of the null hypothesis of iid data. Thus, it seems reasonable to think that the sequence of mean annual temperatures does not come from the model $\mathbf{A}_{t} \stackrel{iid}{\sim} \mathcal{N}(\mu,\,\sigma^{2})$.')
    
    st.markdown('In order to make statistical inferences about the structure of a stochastic process on the basis of an observed record of that process, we must usually make some simplifying assumptions about that structure. The most important such assumption is that of stationarity. Without going into formal definitions, a time series is stationary if it has a constant mean over time and if its variance is also time invariant. Thus, to take these dependencies into account, we will consider a more general model from the time series study. First of all, we will test the stationarity of our time series in order to know which time series model could be applied to our data.')
    
    st.markdown(r'As stated previously in this study, the test we use to test the stationarity of our time series is the Augmented Dickey-Fuller test (ADF). We recall that this test gives us a p-value of $p_{value} = 0.871$, which is far from significant. This result tends to make us think that the time teries is not stationary, which can certainly be explained by the presence of an increasing trend that we had already noticed visually on the plot of the mean temperature. To test this hypothesis of the presence of a tendency, we use another version of the Augmented Dickey-Fuller test to test the trend-stationarity of the time series. In our case, the trend-stationarity of a time serie mean that this time series is stationary if we remove the appropriate polynomial trend from it. Therefore, using this test with the hypothesis of the presence of a linear trend, we obtain a p-value of $p_{value} = 0.016$ which is significant, at the standard significance level of $\alpha = 0.05$ for example. Thus, it seems reasonable to suspect the presence of a trend in the mean temperature.')
    
    
    st.markdown(r'Following this result, we are therefore led to first model the trend of our time series before trying to model our data with a stationary time series model. To model our trend we use a generalized linear regression, i.e. a linear regression in which we do not assume the independence of our errors. We first chose to model the trend as an affine function of time $\mathbf{A^1}_{t} = \beta_{0}^1+\beta_{1}^1t + \epsilon_{t}$ with $\mathbf{\epsilon} = (\epsilon_{1901},...,\epsilon_{2021})^{T}\sim \mathcal{N}(0,\mathbf{\Sigma})$, so as to keep the model simple as possible and to be able to easily infer the sign of $\beta_{1}^1$, which will allow us to detect or not a significant growth of the mean temperature trend. Furthermore, we do not include higher polynomial degree in the trend to avoid obtaining a model with extreme behaviour at these "ends", which we believe is a reasonable restriction given that we are working with average annual temperature data.')
    
    st.markdown(r"To do this, we follow a given procedure for estimate the covariance matrix of $\mathbf{\epsilon}$. This procedure is fully explained below. The following figure shows us the fit of the trend estimate by a line using the GLS. We notice visually that the fit is quite good overall, but that the line has difficulty in approximating the period 1960-1990 correctly. This is due to the fact that, as we saw in the plot of the mean temperature, the mean annual temperature drops sharply in 1962 before resuming a 'normal' behaviour in relation to the rest of the time series.")
    
    with st.expander("Procedure for the estimation of the covariance matrix of the residus"):
        st.markdown(r'''
        For estimate, $\mathbf{\Sigma}$, the covariance matrix of the $\mathbf{\epsilon}$, we want to compute the acf of the time series in order to estimate $\mathbf{\Sigma}$ by $\mathbf{\Sigma}\approx \hat{\sigma}^2\mathbf{T}$, where $\mathbf{T}$ is the Toeplitz matrix generated by the sequence of auto-correlations of our time series. But, as we discuss before, our time series is not stationary and so we normaly cannot compute his autocorrelation sequence. To get around this problem, we use a non-parametric method to make our time series stationary. The method that we use is the LOESS (locally estimated scatterplot smoothing), which is a generalization of moving average and polynomial regression. We iterated this estimation by reducing the number of data used to fit each local polynomial each time, until stationary residuals were obtained. After obtaining a stationary time series, we then have a first estimate of the covariance matrix. With this estimate we are now able to compute a generalized least square (GLS) on our original time series. This GLS regression give us some residus from which, after checking their stationarity, we compute the acf and obtain a new estimation of the covariance matrix as before. Finally, we iterate this last step until convergence of the covariance matrix in the Frobenius norm.
        ''')
    
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
        name = 'GLS without shift'),row = 1, col = 2)
    fig.add_trace(go.Scatter(x = data_Y.Years, y = trendless_gls.f_cts,
        name = r'GLS with shift',
        marker = dict(color = 'mediumvioletred')),row = 1, col = 2)
    
    fig['layout'].update({
        'title': 'Modelisation of the trend by GLS',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'width': 1200,
        'height': 500,
    })
    
    fig.update_yaxes(title_text="Annual mean temperature (°C)", row=1, col=2)
        
    st.plotly_chart(fig)
    
    st.markdown(r'Initially, we wondered whether this sudden change was due to a change in equipment or to the automation of the temperature recordings, which could have affected the average daily temperatures and, by transitivity, the average annual temperatures. But, given the very large temperature difference, this seems unlikely and we have not found any information that would support this hypothesis. However, by doing some additional research we were able to find articles and websites on which the years 1962-1964 were described as particularly cold years with harsh winters ([[5]](http://rodac.canalblog.com/archives/2009/03/30/34397830.html)). This sudden cooling could then be simply due to a temporary local cooling. We then considered this sudden cooling as a rare event, and in order not to affect our study too much, we chose to estimate the trend of the annual mean temperatures as before but this time with a constant shift in the trend since 1962. Mathematically speaking, this leads to the following model $\mathbf{A}_{t}^2 = \beta_{0}^2 + \beta_{1}^2\mathbb{I}(t \geq 1962)  + \beta_{2}^2t + \epsilon_{t}$ with $\mathbf{\epsilon} =  (\epsilon_{1901},...,\epsilon_{2021})^{T}\sim \mathcal{N}(0,\mathbf{\Sigma})$. This led to the estimate GLS estimate on the above figure, where the covariance matrix of the $\mathbf{\epsilon}$ are compute with the same procedure as before.')
    
    st.markdown(r'The results of the two GLS estimates are summarised in the following tables which respectively represent the results of "simple" linear model and the results of the linear model with the addition of a constant from 1962. On the first table, we see that the estimated trend of the time series is $\beta_1^1 = 0.0106$ and that the p-value of the t-test for the covariate $\beta_1^1$ is $p_{value} = 0.001$ under the null hypothesis that $\beta_1^1 = 0$, therefore we have that a 95% confidence interval for $\beta_1^1$  is $[0.005, 0.016]$ and, thus we can reject the fact that $\beta_{1}^1 = 0$ at the significance level $\alpha = 0.05$. This suggests that the trend in the time series is significantly increasing, provided that our model proves to be consistent.')
    
    
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader('GLS Regression Results without constant in 1962')
        HtmlFile = open("DataGenerated/Annual/Annual_summa_without_cts_personaliser", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.markdown(source_code, unsafe_allow_html=True)
    
 
    
    st.markdown(r'The second regression, given on the following table, have estimated the trend of the time series by $\beta_2^2 = 0.0298$ which is larger than the estimate of the previous model. Moreover, under the null hypothesis that $\beta_{2}^2 = 0$, we obtain a p-value of $p_{value} = 1.160e-13$. Thus, we can also reject the null hypothesis. Here again, if our model subsequently proves to be consistent with our data, then we would have a significant indication of an increasing trend in mean temperatures.')
        
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader('GLS Regression Results with constant in 1962')
        HtmlFile = open("DataGenerated/Annual/Annual_summa_with_cts_personaliser", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.markdown(source_code, unsafe_allow_html=True)
    
    st.markdown(r'''Even if the regression with a shift seems visually to better model the trend of our time series, we have a priori no reason to choose one model over the other. To find out if our second model brings a real advantage in the modelling we have performed a loglikelihood ratio test between our two nested models. In our case, the loglikelihood-ratio test allow us to compare the two model and to test if the addition of the shift in the second model bring some relevant information on the data or not. Formally speaking we will test the null hypothesis $\beta_{1}^1 = 0$. This loglikelihood test give us a p-value of $p_{value} = 4.701e-07$ which allow us to reject the null hypothesis at a standard significante level of $\alpha = 0.05$. Thus, given that result, we are led to consider the second model since the loglikelihood-ratio make us reject the hypothesis that $\beta_{1}^1 = 0$.
        ''')
    
    st.markdown(r'From now we choose to consider the second model for estimate the trend, the one with the shift in 1962. This choice is justified by the result of the previous loglikelihood test and by the think that the sudden drop in temperature in 1962 is the result of a rare event, which we believe should be removed from the data so as not to influence our simple trend estimate. As this sudden drop in temperature is estimated at $\beta_1^1 = -1.522 (\degree C)$ by the second regression model, it would be relevant to study this phenomenon in more details in future analyses. Therefore, in the following of this study we will consider the new time series $\{\mathbf{\tilde{A}}_{t}\}_{t=1901}^{2021}$ which is the time series $\{\mathbf{A}_{t}\}_{t=1901}^{2021}$ from which we have subtracted the trend calculated above. The figure below allows us to visualise this new time series.')
    
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
    
    st.markdown(r'''Now, it is interesting to test again the independence of the elements of the new time series. Especially since this new time series is now stationary. Indeed, the Augmented Dickey-Fuller test on this one gives us a p-value of $p_{value} = 2.081e-08$ and thus allows us to reject the null hypothesis of non-stationarity at a significance level of $\alpha = 0.05$. To test the independence of the time series $\{\mathbf{\tilde{A}}_{t}\}_{t=1901}^{2021}$, we proceed as before, we have calculated the auto-correlation and partial auto-correlation sequences of the time series, which can be seen in the figure below. Almost all terms of the acf are within the approximate confidence interval in blue, but one can see that some terms are still outside. Moreover, we see on the other plot that the p-values associated with the Ljung-Box tests allow us to reject the null hypothesis of independence of the time series elements, even if the McLeod-Li test allow us to reject the null hypothesis only for few lags.
        ''')
    
    st.image("figure/Detrended_mean_acf_pacf_cst.png")
    
    pval_ind2 =pd.read_csv(
    "DataGenerated/Annual/Annual2_pvalue_indep_GLS")
    
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
    fig.add_trace(go.Scatter(x=pval_ind2.index+1, y=pval_ind2.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li',
                    marker=dict(size=8)),row =1, col = 2)
    #fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'P-values of the Ljung-Box and McLeod-Li tests on the trendless mean temperature',
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
    
    st.markdown(r'Even though removing the trend from the time series of mean annual temperatures has reduced the dependence between observations, there is still enough dependence between observations to use a times series model to model $\{\mathbf{\tilde{A}}_{t}\}_{t=1901}^{2021}$. ')
    
    st.markdown(r'Then, we now have choose a good time series model to model our time series $\{\mathbf{\tilde{A}}_{t}\}_{t=1901}^{2021}$.The following figure allows us to compare several models for the time series thanks to the Akaike information criterion (AIC). We use the AIC in order to make a model selection since it is an estimator of prediction error and thereby relative quality of statistical models for our set of data. We have tried to model the time series as arising from an $ARMA(p,q)$ for $p\in\{0,...,3\}$ and $q\in\{0,...,9\}$. Our choice of restricting $p$ and $q$ is mainly due to the fact that our time series is not very large and that we wanted to try to keep the model as simple as possible to model our data. To estimate the coefficients of a given model, we use the method "ARIMA" in the $\textbf{statsmodels}$ packages on python.')
    
    aic =pd.read_csv(
    "DataGenerated/Annual/Annual_modelSelection.csv")
    
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

    st.markdown(r'By choosing the model which minimises the AIC, we are led to consider the model $ARMA(2,0)$ to model the time series $\{\mathbf{\tilde{A}}_{t}\}_{t=1901}^{2021}$, which amounts to considering a model $AR(2)$. On the following plot, we see the residus of the estimation of the trendless time series as a $AR(2)$ process, given by $\mathbf{\tilde{A}_{t}} = \hat{\phi_1} \mathbf{\tilde{A}_{t-1}} + \hat{\phi_2}\mathbf{\tilde{A}_{t-2}} + \mathbf{\tilde{\epsilon}_{t}}$ with $\mathbf{\tilde{\epsilon}_{t}}\sim \mathcal{N}(0,\mathbf{\hat{\sigma^2}})$, where the "ARIMA" method estimate the coefficient with $\hat{\phi}_1 = 0.206260, \hat{\phi}_2 = 0.223250 \text{ and } \hat{\sigma}^2 = 0.258059$ ')
    
    resid_arma2_0 =pd.read_csv("DataGenerated/Annual/Annual_resid_ARMA2_0.csv")
    
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
    
    st.markdown(r'In order to confirm the consistency of our model, we tested the independence and the distribution of the residuals obtained. The following figure show us the result of the Ljung-Box and McLeod-Li test on the residuals of our model, we see that the results do not allow us to reject the null hypothesis of independence of the tests, at a significance level of $\alpha = 0.05$. Thus, the $AR(2)$ model seems to capture well the dependence structure between the elements of the time series $\{\mathbf{\tilde{A}}_{t}\}_{t=1901}^{2021}$.')
    
    pval_ind3 =pd.read_csv(
    "DataGenerated/Annual/Annual2_pvalue_indep_ARMA2_0")
    
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
    fig.add_trace(go.Scatter(x=pval_ind3.index+1, y=pval_ind3.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li',
                    marker=dict(size=8)),row =1, col = 2)
    #fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'P-values of the Ljung-Box and McLeod-Li tests on the residus of the ARMA(2,0)',
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
    
    st.markdown(r'We therefore seem to have obtained residuals that are independent of each other. Moreover, after performing the Jarque-Bera test, which has the null hypothesis that our data are from a normal distribution, and the Goldfeld-Quandt test, which has the null hypothesis that the data are homoskedastic, we get the p-values : ')
    
    
    col1,col2,col3 = st.columns([1,1,1])
    
    with col2:
        st.latex(r'''\text{Normality test : }p_{value} = 0.725 \\
                \text{Heteroskedasticity test : }p_{value} = 0.157
        ''')
        
    with st.expander("Jarque-Bera and Goldfeld_Quandt tests"):
        st.markdown(r'''$\textbf{The Jarques-Bera test}$ :  is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution. The test statistic is defined as : $JB = \frac{n}{6} \left( S^2 + \frac{1}{4} \left(K-3\right)^2 \right)$, where $S =\frac{ \frac{1}{n} \sum_{i=1}^{n} (x_i-\bar{x})^3 }{ \left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^{\frac{3}{2}} }$ and $K =\frac{ \frac{1}{n} \sum_{i=1}^{n} (x_i-\bar{x})^4 }{ \left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^{2} }$.\\
            If the data comes from a normal distribution, the JB statistic asymptotically has a chi-squared distribution with two degrees of freedom, so the statistic can be used to test the hypothesis that the data are from a normal distribution. The null hypothesis is a joint hypothesis of the skewness being zero and the excess kurtosis being zero.''')
        st.markdown(r'''$\textbf{The Goldfeld-Quandt tests}$ : is accomplished by undertaking separate least squares analyses on two subsets of the original dataset: these subsets are specified so that the observations for which the pre-identified explanatory variable takes the lowest values are in one subset, with higher values in the other. The subsets need not be of equal size, nor contain all the observations between them. The test assumes that the errors have a normal distribution. There is an additional assumption here, that the design matrices for the two subsets of data are both of full rank. The test statistic used is the ratio of the mean square residual errors for the regressions on the two subsets. This test statistic corresponds to an F-test of equality of variances.
            ''')
    
    st.markdown(r'Both p-values are not significant at a threshold of $\alpha = 0.05$, so we cannot reject the null hypotheses of normality and homoscedasticity. Therefore, it seems that the model we develop for the original time series of mean temperature is appropriate, in the sense that the trend that we estimate with the GLS is ')
    
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.title("Discussion of the Results")
    
    st.markdown(r"In this preliminary part of our study on the evolution of the mean temperature trend, we have restricted our analyses to the mean annual temperatures at the Geneva observatory. This allowed us to obtain initial results regarding the increase in the annual trend of mean temperatures at Geneva. We found the presence of an increase in the annual trend of the mean temeprature. Indeed, we estiamte the trend as a linear trend with a estimated slope of $\beta_2^2 = 0.0298$ in the 95% confidence interval $[0.023, 0.037]$. However, we should not forgot that we model our data in order to remove the sudden drop of the mean temeprature in 1962. Therefore, even though we estimated the slope of the linear trend to be $\beta_2^2 = 0.0298$ which implies a strong increase in the annual temperature trend, we do not know if this sudden drop in temperature is an isolated event or if it is a phenomenon that is likely to be repeated over a longer period than the one we are studying, which would question our result regarding the increase in the trend. Moreover, it is interesting to note that our estimate is in the same order of magnitude as the one reported by the article of the city of Geneva (see [[2]](https://www.geneve.ch/fr/actualites/dossiers-information/changement-climatique-geneve/comprendre/effets-suisse-geneve#)) ")


    
    
    st_forecasting()
#########################################################################################


def introduction():

    st.markdown('Hey there! This app is developped in the context of a project supervised by Prof. [Mehdi Gholam](https://people.epfl.ch/mehdi.gholam?lang=fr) at EPFL, in the context of the course *Statistical Computation and Visualization*.')

    
    st.header("Introduction")

    st.markdown("Statistical and visual analysis of data are major components of the general data science domain. In this project, we look at various datasets revolving around meteorological recordings from various stations within Switzerland. Statistical analysis of meteorological data plays an important role in understanding and modeling key features in climate change, as well as making short to long-term predictions on certain meteorological elements. Here, we are interested in providing an efficient pipeline aiming at analysing meteorological data through basic statistical methods such as linear regression and time series analysis. Moreover, we concentrate in providing a significant amount of visualization tools to combine with the statistical results.")

    st.markdown("The main question we try to answer revolves around the mean temperature element, recorded across Switzerland. We formulate it as follows.")

    st.markdown("*Question.* Is there a significant increase in the average temperature trend in Switzerland from 1901 to the present day?")

    st.markdown("To try to answer this question, we will first focus on the evolution of the average temperature in Geneva. This will allow us to refine and improve our statistical study on the Geneva observatory data, before extending it to other weather stations. The main mathematical objects of our study are time-series of mean temperatures, which we create along various conventions (*i.e.* as weekly, monthly and yearly averages). This resembles the approach seen in [[1]](https://www.researchgate.net/publication/271306657_Statistical_Analysis_from_Time_Series_Related_to_Climate_Data), where so-called *climate-indices* are computed on different time-range averages, based on restrictions on the proportion of missing values within the data.")
    
    st.markdown("Analysis of meteorological data plays an important role within the scientific community, and we refer to various articles and papers regarding the way we formulate our questions and study the data. Our work is motivated by the inference (see [[2]](https://www.geneve.ch/fr/actualites/dossiers-information/changement-climatique-geneve/comprendre/effets-suisse-geneve#)) that Geneva is one the towns in the world whose increase in temperature during the period 2020-2029 is the highest, being of approximately 2.5°C. Although our analysis is based on mean temperature recordings, we give the possibility to observe and interact with data concerning sunshine duration records as well with the data visualization option. This is motivated by the fact that sunshine duration presents a behavior similar to mean temperature in the sense that it heavily increased in the past 40 years (see [[3]](https://www.ge.ch/statistique/tel/publications/2006/analyses/coup_doeil/an-co-2006-26.pdf)).")


    st.write("""The entirety of our work is contained in this *Streamlit* web application, structured as follows. There is a left sidebar on which you can find the link to our GitHub repository, as well as our contacts (GitHub and EPFL profiles). Now, the sidebar also contains a list of main options to choose from, as follows.""")
    st.write(
        """
    - Project description
    - Time-series analysis
    - Data visualization
    """
    )


    st.subheader("Project description")

    st.write("""

With this option, you can find the global description of our project, quite equivalent to a $\textit{Readme}$ file on GitHub. We introduce the subject, the context and the aims of our study.""")

    st.subheader("Time series analysis")

    st.write("""Here, you can read about our statistical analysis in more details and have access to the statistical results of the study. This essentially encapsulates a final report, as well as a good way of observing plots and figure, as they are produced with the $\textit{Plotly}$ library. The deeper statistical study heavily relies on time-series analysis and forecasting models, for which we notably refer to the book [[4]](https://link.springer.com/book/10.1007/978-3-319-29854-2). We also implement a part of our linear regression methods based on a to-Python translation of the R tutorial [[5]](https://lbelzile.github.io/lineaRmodels/).""")

    st.subheader("Data visualization")

    st.write("""Finally, you can choose this option to interact with the data directly. There are four main visualization features implemented here, as follows. The first three features aim at providing a summary of descriptive statistics, while the fourth one offers a way of interacting with an annual forecasting model.""")
    
    st.write(
        """
    - Display time-slider summary
    - Display time-window feature
    - Display correlation network feature
    """
    )



    st.markdown("""We make the data interaction possible through checkboxes, sliders and other widget options.""")


    
