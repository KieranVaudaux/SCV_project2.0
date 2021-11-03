from ipywidgets import fixed, interact, interact_manual, interactive
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st


def display_date_slider(years):
    #start_date = min(years)
    #end_date = max(years)
    #step_time = 1
    # the first window delta
    #init_delta = 5
    #start_date, end_date = st.sidebar.slider('Start time', start_date, end_date, value=(start_date),
    #                                        step=step_time, help="This is where you can chose the start time")
    
    year = st.sidebar.slider('Year', int(min(years)), int(max(years)-1), 1960)
    return year