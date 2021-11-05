import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
#from pyvis.network import Network
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib
import base64

from streamlit_functions import display_date_slider, st_display_pdf, main
from visual_features import *



st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: center;'> Meteorological data visualization</h1>",
            unsafe_allow_html=True)


MODES = ['Descriptive Statistics', 'Time Series Visualization', 'Time Series Analysis']

st.sidebar.header('Options')

INFO = st.sidebar.radio("Content",('Project description', 'Whole study', 'Data Visualization'))
    

if INFO == 'Project description':
    
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
    
    #st.header('Contact')
    
    #st.markdown('We are students in the [mathematics department](https://www.epfl.ch/schools/sb/research/math/fr/) at EPFL.')
    #st.markdown('If you happen to have any question or comment on this project, we are happy to answer - you can find our #contacts in the sidebar.')
    

elif INFO == 'Whole study':
    
    main()
    
    with open("../reports/SCV_report.pdf", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download report",
            data=file,
            file_name="SCV_report.pdf",
            mime="report/pdf"
        )
    
    
    
    
elif INFO == 'Data Visualization':

    SELECTED_MODE = st.sidebar.selectbox("Visualization mode", MODES, index=0)
    if SELECTED_MODE == MODES[0]:
        
        elt = st.radio("What do you want to observe?",('Mean temperature', 'Sunshine duration'))
        ######################################
        # (1) Visualization
        ######################################
        st.header("Descriptive Statistics - Interactive Visualization")
        st.sidebar.header("Descriptive Data visualization")
        
        
        if elt == 'Mean temperature':

            df = pd.read_table('../data/observatoire-geneve/TG_STAID000241.txt', sep = ',',
                                        names = ['SOUID','DATE','TG','Q_TG'], skiprows = range(0,20))

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

        elif elt == 'Sunshine duration':
            
            df = pd.read_table('../data/SS_STAID000241.txt', sep = ',',
                                    names = ['SOUID','DATE','TG','Q_TG'], skiprows = range(0,20))

            df['Year'] = [int(str(d)[:4]) for d in df.DATE]
            df['Month'] = [int(str(d)[4:6]) for d in df.DATE]
            df['Day'] = [int(str(d)[6:8]) for d in df.DATE]

            #Compute the day of the year for each year
            day_of_year_s = np.array(len(df['Day']))

            adate_s = [datetime.strptime(str(date),"%Y%m%d") for date in df['DATE']]
            df['Day_of_year'] = [d.timetuple().tm_yday for d in adate_s]
            
            years = df.Year.unique()

        years = np.delete(years, years.shape[0]-1)
        year = display_date_slider(years)
        
        bins = [5*i for i in range(10,21)]
        bin_ = st.sidebar.slider('Bins in histogram', 50, 100, 75)

        #figs, ax = plt.subplots(1)
        #pie_chart_missing(year, ax, df)


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
        #right_column.pyplot(fig6, figsize=(8, 10))
        
        if elt == 'Mean temperature':
            
            fig7, ax7 = plt.subplots(1)
            plot_mean_temp_global(years, x=list(years).index(year), ax=ax7, df_av=df_av, element=elt)
            right_column.pyplot(fig7, figsize=(10, 10))
        




        # Temporal cursor

        #plot_stats_window(years,df,"Mean temperature")



    #elif SELECTED_MODE == MODES[1]:

    elif SELECTED_MODE == MODES[2]:
        
        st.header("Time Series Analysis - Main Results")
        st.sidebar.header("Time Series Statistical Analysis")
        
        elt = st.selectbox('Which result do you want to see?', ('p-values of the Ljung Box, Box-Pierce test, and McLeod-Li', 
                                             'ACF and PACF',
                                             'QQ-plot',
                                             'p-values of the Mann-Whitney U test'))

        
        if elt == "p-values of the Ljung Box, Box-Pierce test, and McLeod-Li":
            
            st.markdown("#### p-values of the Ljung Box, Box-Pierce test, and McLeod-Li")
            st.image("p_values_ljung.png")
            
            with st.expander("See explanation"):
                st.write("""A nice description wow guys streamlit is so cool.
                """)
            
            
            
        elif elt == "ACF and PACF":
            
            st.markdown("#### ACF and PACF")
            st.image("ACF_and_PACF.png")
            
            with st.expander("See explanation"):
                st.write("""A nice description wow guys streamlit is so cool.
                """)
                
            
        elif elt == "QQ-plot":
            
            st.markdown("#### QQ-plot")
            st.image("QQ_plot.png")
            
            with st.expander("See explanation"):
                st.write("""A nice description wow guys streamlit is so cool.
                """)
            
        elif elt == "p-values of the Mann-Whitney U test":
            
            st.markdown("#### p-values of the Mann-Whitney U test")
            st.image("mann_whitney.png")
            
            with st.expander("See explanation"):
                st.write("""A nice description wow guys streamlit is so cool.
                """)
            
        

if st.sidebar.button("GitHub"):

    st.sidebar.markdown("The entire code of the project, from source code to notebooks, is available at our GitHub repo [here](https://github.com/LucaNyckees/SCV_project1). Have a look!")

if st.sidebar.button("Contacts"):
    
    st.sidebar.markdown("""
        * Luca Bracone ([EPFL](https://people.epfl.ch/luca.bracone), [GitHub](https://github.com/jkasalt))\n
        * Luca Nyckees ([EPFL](https://people.epfl.ch/luca.nyckees), [GitHub](https://github.com/LucaNyckees))\n
        * Blerton Rashiti [EPFL](https://people.epfl.ch/blerton.rashiti), [GitHub](https://github.com/BlertonRashiti))\n
        * Kieran Vaudaux [EPFL](https://people.epfl.ch/kieran.vaudaux), [GitHub](https://github.com/KieranVaudaux)) 
        """)
    

                