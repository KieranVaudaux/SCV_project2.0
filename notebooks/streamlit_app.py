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
from paths import paths

from streamlit_functions import *


# STREAMLIT STRUCTURAL CONFIGURATION

st.set_page_config(layout='wide')

#-----


#-----

st.markdown("<h1 style='text-align: center;'> Meteorological data visualization</h1>",
            unsafe_allow_html=True)


MODES = ['Descriptive Statistics','Time Series Analysis', 'Time Series Visualization', 'Time Series Analysis']
MODES_TS = ["Evolution of the Mean Temperature at Geneva Observatory","Analysis of annual Mean Temperature at Geneva Observatory","Analysis of monthly Mean Temperature at Geneva Observatory",'Description of statistical tools']

st.sidebar.header('Options')

INFO = st.sidebar.radio("Content",('Project description', 'Evolution of the Mean Temperature at Geneva Observatory','Whole study', 'Data Visualization'))
    

# PROJECT DESCRIPTION

if INFO == 'Project description':
    
    description()
    
# Written analysis with time series
elif INFO == "Evolution of the Mean Temperature at Geneva Observatory":
    
    SELECTED_MODE = st.sidebar.selectbox("Part", MODES_TS, index=0)
    if SELECTED_MODE == MODES_TS[0]:
        annual_intro()
    elif SELECTED_MODE == MODES_TS[1]:
        annual_analysis()
    elif SELECTED_MODE == MODES_TS[2]:
        description()
    elif SELECTED_MODE == MODES_TS[3]:
        description()
# DISPLAYING PDF REPORT
    
elif INFO == 'Whole study':
    
    main()
    
    with open("../reports/SCV_report.pdf", "rb") as file:
        btn = st.sidebar.download_button(
            label="Download report",
            data=file,
            file_name="SCV_report.pdf",
            mime="report/pdf"
        )
    
    
# INTERACTIVE AND STATIC DATA VISUALIZATION

elif INFO == 'Data Visualization':

    
    
    SELECTED_MODE = st.sidebar.selectbox("Visualization mode", MODES, index=0)
    if SELECTED_MODE == MODES[0]:
        
        elt = st.radio("What do you want to observe?",('Mean temperature', 'Sunshine duration'))
        
        st.header("Descriptive Statistics - Interactive Visualization")
        st.sidebar.header("Descriptive Data visualization")
        
        path = paths(elt)
        
        
        df = pd.read_table(path, sep = ',', names = ['SOUID','DATE','TG','Q_TG'], skiprows = range(0,20))

        plot_stats_window_st(df,elt)


    #elif SELECTED_MODE == MODES[1]:

    elif SELECTED_MODE == MODES[2]:
        
        results_display()
            
        
# SIDEBAR BONUS OPTIONS

if st.sidebar.button("GitHub"):

    github()

if st.sidebar.button("Contacts"):
    
    contacts()
    

                
