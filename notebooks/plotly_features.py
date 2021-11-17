import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


colors_list = px.colors.qualitative.Plotly



def layout(fig):
    
    fig['layout'].update({
        'showlegend': True,
        'width': 600,
        'height': 500,
    })
    
def plotly_mean_temp(year, fig, df, elt):
    
    data=df[df.Year==year]

    fig.add_trace(go.Scatter(x=data['Day_of_year'], y=data['TG'],
            fill=None,
            mode='lines',
            name = year
            #line_color='rgb(184, 247, 212)',
            ))
    fig.update_layout(
    title=elt+" curve",
    xaxis_title="day of the year",
    yaxis_title=elt
    )
    layout(fig)
    
    
def plotly_hist_mean(year, fig, df, elt, bins, iterate=False):
    
    data=df[df.Year==year]['TG']
    
    if iterate:
        
        fig.add_trace(go.Histogram(x=data, 
                                   name=str(year),
                               xbins=dict( 
        start=-130.0,
        end=280,
        size=410/float(bins)
    )))
        
    else:
    
        fig.add_trace(go.Histogram(x=data, 
                                   name=str(year),
                                   marker_color=colors_list[5],
                                   xbins=dict( 
            start=-130.0,
            end=280,
            size=410/float(bins)
        )))
    
    fig.update_layout(
    title=elt+" histogram",
    xaxis_title="value",
    yaxis_title="count"
    )
    layout(fig)
    
    
def plotly_pie_chart_missing(year, fig, df):
    
    fig.update_layout(
    title="Proportion of missing values",
    )
    layout(fig)
    
def plotly_min(years, x, fig, df, elt):
    
    fig.update_layout(
    title="Minimum " + elt + " curve",
    xaxis_title="year",
    yaxis_title="Minimum " + elt
    )
    layout(fig)
    
    year = years[x]

    min_temps = [min(df[df.Year==year]['TG']) for year in years]
    
    fig.add_trace(go.Scatter(x=years, y=min_temps,
            fill=None,
            name="",
            mode='lines',
            line_color=colors_list[7]
            ))
    fig.add_trace(go.Scatter(x=[years[x]], y=[min_temps[x]],
            fill=None,
            mode='markers',
            name = str(year)
            ))
    layout(fig)
    
def plotly_max(years, x, fig, df, elt):
    
    fig.update_layout(
    title="Maximum " + elt + " curve",
    xaxis_title="year",
    yaxis_title="Maximum " + elt
    )
    
    year = years[x]

    max_temps = [max(df[df.Year==year]['TG']) for year in years]
    
    fig.add_trace(go.Scatter(x=years, y=max_temps,
            fill=None,
            name="",
            mode='lines',
            line_color='rgb(184, 247, 212)'
            ))
    fig.add_trace(go.Scatter(x=[years[x]], y=[max_temps[x]],
            fill=None,
            mode='markers',
            name = str(year)
            ))
    layout(fig)
    
    
def plotly_std(years, x, fig, df):
    
    fig.update_layout(
        title="Standard deviations curve",
        xaxis_title="year",
        yaxis_title="std"
        )
    
    year = years[x]
    stds = [df[df.Year==year]['TG'].std() for year in years]
    
    fig.add_trace(go.Scatter(x=years, y=stds,
            fill=None,
            mode='lines',
            name="",
            line_color='violet'
            ))
    fig.add_trace(go.Scatter(x=[years[x]], y=[stds[x]],
            fill=None,
            mode='markers',
            name = str(year)
            ))
    layout(fig)
    
    
def plotly_mean_temp_global(years, x, fig, df_av, element):
    
    fig.update_layout(
    title="Average " + element + " curve",
    xaxis_title="year",
    yaxis_title="Average " + element
    )
    layout(fig)
    
    year = years[x]
    temps = [df_av[df_av.Year==year]['ATG'] for year in years]

    fig.add_trace(go.Scatter(x=df_av['Year'], y=df_av['ATG'],
            fill=None,
            mode='lines',
            name="",
            line_color='cornflowerblue'
            ))
    fig.add_trace(go.Scatter(x=[years[x]], y=[temps[x]],
            fill=None,
            mode='markers',
            name = str(year)
            ))