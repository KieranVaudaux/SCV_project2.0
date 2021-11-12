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

    
