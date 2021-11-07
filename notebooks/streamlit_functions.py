from ipywidgets import fixed, interact, interact_manual, interactive
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import base64


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
    
    #st_display_pdf("../reports/Statistical_analysis_and_visualization_of_meteorological_data-5.pdf")
