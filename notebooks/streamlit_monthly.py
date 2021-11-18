import streamlit as st
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
import calendar
from TimeSerie_fct import create_monthly_avg_time_serie
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, pacf
from sklearn.utils import resample
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def monthly_analysis():
    c1,c2,c3 = st.columns([1,10,1])
    with c2:
        st.title("Monthly Mean Temperature at Geneva Observatory")
    
    data_M = pd.read_csv("DataGenerated/Monthly/Monthly_Mean.csv")

    
    st.markdown(r"In the previous analysis, we briefly questioned the relevance of considering the average annual temperatures in Geneva, given that during a year the range of observed temperatures is quite large. However, we are interested here in modelling the average monthly temperatures observed in Geneva Observatory, which makes these concerns less relevant since the temperatures in any one month are all much more similar.")
    st.markdown(r"In this analysis, we will try to model the time series of monthly average temperature at the Geneva Observatory. Through this model we will try to see if we can detect a significant increase in the mean temperature trend as we have been able to do with the annual means. To begin, we can see on the following figure the interactive plot of the entire time serie of the monthly mean temperature. The interactive part of the plot is the window on which the time series is plotted, we can set the window to a given size and scroll it.")
    
    fig = px.line(data_M, x='Date', y='Mean')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6 months", step="month", stepmode="backward"),
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(count=10, label="10 years", step="year", stepmode="backward"),
                dict(count=30, label="30 years", step="year", stepmode="backward"),
                dict(count=50, label="50 years", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    fig['layout'].update({
        'title': 'Time Series of the monthly mean temperatures at Geneva Observatory',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'yaxis': {
            'title':"Temperature (°C)"
        },
        
        'width': 1200,
        'height': 500,
        'legend':{
            'title':""
                         }
    })
   
    st.plotly_chart(fig)
    
    

    def create_button(label, n_visible):
        return dict(label=label,
                         method="update",
                         args=[{"visible": [bool(n_visible==m) for m in np.arange(1,13)]}
                         ])
    
    st.markdown("On the full plot of the time serie, the seasonal component is clearly visible and it can be difficult to detect any global behaviour of a possible trend. In order to avoid this disadvantage we can look at the figure above, where we plot the time series of mean temperature by month on the period 1901-2021. This figure allow us to have a better visualization of the tendancy of the time serie visualisation of the time series trend through each month.")
    
    fig = go.Figure()
    
    for i in range(1,13):
        fig.add_trace(
            go.Scatter(x=data_M[data_M.Month == float(i)].Years,
                       y=data_M[data_M.Month == float(i)].Mean,
                       name=calendar.month_name[int(i)],
                       line=dict(color=px.colors.qualitative.G10[(i-1)%9])))
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.87,
                y=1.12,
                buttons=list([create_button(calendar.month_name[int(m)],m) for m in np.arange(1,13)]),
            )
        ])
    fig.update_layout(
    title_text="Time series of the mean temperature by month", title_x= 0.5, title_y= 1,
    xaxis_domain=[0.05, 1.0]
    )
    
    fig['layout'].update({
        'xaxis': {
            'title': 'Years',
            'zeroline': False
        },
        'yaxis': {
            'title':"Mean temperature (°C)"
        },
        
        'width': 1200,
        'height': 600
    })
        
    st.plotly_chart(fig)
    
    st.markdown("On the plots above a slight increase in the trend can be seen, but it is much less pronounced than in the time series of annual averages. Similarly, in the figure below we plot the standard temperature anomalies by month, but again it is more difficult to detect an increase in the trend.")
    
    fig = go.Figure()
    

    for i in range(1,13):
        mean = data_M[data_M.Month == float(i)].Mean.mean()
        sigma = data_M[data_M.Month == float(i)].Mean.std()
        
        fig.add_trace(
            go.Bar(x=data_M[data_M.Month == float(i)].Years,
                       y=(data_M[data_M.Month == float(i)].Mean-mean)/sigma,
                       name=calendar.month_name[int(i)],
                       marker=dict(color=px.colors.qualitative.Dark24[i])))
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.87,
                y=1.12,
                buttons=list([create_button(calendar.month_name[int(m)],m) for m in np.arange(1,13)]),
            )
        ])
    fig.update_layout(
    title_text="Monthly standardized temperature anomalies", title_x= 0.5, title_y= 1,
    xaxis_domain=[0.05, 1.0]
    )
    
    fig['layout'].update({
        'xaxis': {
            'title': 'Years',
            'zeroline': False
        },
        'yaxis': {
            'title':"Standardized  anomalies"
        },
        
        'width': 1200,
        'height': 600
    })
        
    st.plotly_chart(fig)
    
    with st.expander("Standardised anomalies explanation"):
        st.markdown(r'The temperature anomaly is the difference between the temperature measured in a place (here the mean temperature at Geneva), compared to the normal average temperature observed in this same place over a certain period (here we look at the period 1901-2021). On a given period, the standardised anomalies are computed by normalizing the anomalies by the standard deviation of the temperature from the mean temperature of this period.')

    st.markdown(r"To model the time series of average temperatures, we will this time directly model our time series using a generalized linear regression. This is all the more justified given the seasonal component that we have observed on the time series plot. Let denote, for the rest of this study, the monthly time serie by $\mathbf{M}_{t,m_t}$ for $t = 1901, ..., 2021$ , and $m_t = 1,...,12$ for $t = 1901,...2020$ and $m_{2021} = 1,...,8$.")

    st.header("A HEADER ??")
    
    st.markdown(r"The first model we will consider for model the time series $\mathbf{M}_{t,m_t}$ is the following : $\mathbf{M}_{t,m_t} = \sum_{i=1}^{12} \beta_i \mathbb{I}(m_t = i) + \beta_{13}(t-1900) + \epsilon_{t,m_t}$ with $\mathbf{\epsilon} \sim \mathcal{N}(0,\mathbf{\Sigma})$ with a unknown covariance matrix $\mathbf{\Sigma}$. This model is equivalent to suppose that the underlying structure of the process from which the time series of mean temperatures is drawn, is given by a different average temperature for each month over which there is a linear increase in these average temperatures from year to year, with the addition of a gaussian perturbation which could be correlated from month to month. Futhermore, we have chosen such a model in order to be able to characterize and quantify in a simple way a possible increase of the monthly average temperature trend. Indeed, a significant increase in the trend would result in a significantly positive coefficient $\beta_13$ in our model, if our model proves to be plausible. This would not have been so simple if we had wanted to estimate the trend and seasonality using a non-parametric method.")
    
    st.markdown(r"In order to estimate the parameters $\beta_1, ..., \beta_{13}$ we will use a generalized least square (GLS) regression. As in the case of the annual mean temperature, we will need to estimate  the covariance matrix $\mathbf{\Sigma}$ by following a certain procedure (more details on the procedure are available below). On the interactive figure below, we can see the result of the GLS estimation of the model and we could compare the result with the monthly mean time series. In order to better visualize the result, we advise you to compare the two time series on a 10 or 30 years rolling window.")
    
    with st.expander("Procedure for the estimation of the covariance matrix of the residus"):
        st.markdown(r"For estimate the covariance matrix, we will process similarly to the annual case. As the time series is clearly not stationary given its seasonal component, we will first make it stationary by using a non-parametric method to remove the seasonal component and the possible trend from the time series. The method we will use to make the time series stationary is an STL method (Seasonal-Trend decomposition using LOESS) implemented in the python package $\textbf{statsmodels}$. Then, after checking that the residus of the estimation are stationary with the Augmented Dickey-Fuller test, thus we can compute the acf of the time serie of the residus. Then, we make a first estimation of $\mathbf{\Sigma}$ by $\mathbf{\Sigma} \approx \hat{\sigma}^2\mathbf{T}$, where $\mathbf{T}$ is the Toeplitz matrix generated by the sequence of auto-correlations of our time series and $\hat{\sigma}^2$ is the sample variance of the residus. With this estimate we are now able to compute a generalized least square (GLS) on our original time series. This GLS regression give us some residus from which, after checking their stationarity, we compute the acf and obtain a new estimation of the covariance matrix as before. Finally, we iterate this last step until convergence of the covariance matrix in the Frobenius norm.")
    
    monthly_seasonless_gls = pd.read_csv("DataGenerated/Monthly/Monthly_Seasonless.csv")
    
    fig = px.line(monthly_seasonless_gls, x='Date', y='f')
    
    fig = go.Figure()
                    
    fig.add_trace(
        go.Scatter(x=data_M.Date,
                    y=data_M.Mean,
                    name="Monthly mean",
                    line=dict(color=px.colors.qualitative.Plotly[0])))
    fig.add_trace(
        go.Scatter(x=monthly_seasonless_gls.Date,
                    y=monthly_seasonless_gls.f,
                    name="Seasonnal estimation",
                    line=dict(color=px.colors.qualitative.Plotly[1])))
                    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.87,
                y=1.12,
                buttons=list([dict(label="Seasonnal estimation",
                         method="update",
                         args=[{"visible": [False, True]}
                         ]),
                        dict(label="Monthly mean and seasonnal estimate",
                         method="update",
                         args=[{"visible": [True, True]}
                         ])
                    ]),
            )
        ])
    fig.update_layout(
    title_text="Time series of the mean temperature by month", title_x= 0.5, title_y= 1,
    xaxis_domain=[0.05, 1.0]
    )
    
    fig['layout'].update({
        'xaxis': {
            'title': 'Years',
            'zeroline': False
        },
        'yaxis': {
            'title':"Mean temperature (°C)"
        },
        
        'width': 1200,
        'height': 600
    })

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6 months", step="month", stepmode="backward"),
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(count=10, label="10 years", step="year", stepmode="backward"),
                dict(count=30, label="30 years", step="year", stepmode="backward"),
                dict(count=50, label="50 years", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig)
    
    st.markdown("By visually analysing the plots above, we can see that our model seems to be rather good in terms of fit with the time series of average monthly temperatures. Moreover, when we only look at the estimation, we clearly see a augmentation of the trend. To study this in more detail, we have below the generalized regression result. ")
    
    col1, col2, col3= st.columns([1,1,1])
    with col2:
        st.subheader('GLS Regression Results')
    col1, col2= st.columns([1,1])
    with col1:
        HtmlFile = open("DataGenerated/Monthly/summary1_GLS_personaliser", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.markdown(source_code, unsafe_allow_html=True)
        
    with col2:
        HtmlFile = open("DataGenerated/Monthly/summary1_GLS_personaliser1", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.markdown(source_code, unsafe_allow_html=True)
    
    st.markdown(r"On the above table, we see the all estimates of the coefficients $\beta_1, ..., \beta_{12}$ which are given by $x1, ..., x12$ on the rigth table. However, as the coefficient which represent the behaviour of the trend is $\beta_{13}$, which corresponds in the right table to $x13$, we are mainly interested in the estimation of the latter. The GLS gives us the estimation $\beta_{13} = 0.0103$ with a 95% confidence interval given by $[0.005, 0.016]$. If our model subsequently proves to be appropriate, this confidence interval for $\beta_{13}$ would be an argument for the presence of a significant increase in the trend of monthly mean temperatures.")
    
    st.markdown(r"From now, we will consider the new time series $\{\mathbf{\tilde{M}}_{t,m_t}\}$, which is the time series of average temperatures from which the estimated seasonal component and the trend have been removed. On the the figure below we coul visualize this new timw series.")
    
    fig = px.line(monthly_seasonless_gls, x='Date', y='resid')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6 months", step="month", stepmode="backward"),
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(count=10, label="10 years", step="year", stepmode="backward"),
                dict(count=30, label="30 years", step="year", stepmode="backward"),
                dict(count=50, label="50 years", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    fig['layout'].update({
        'title': 'Time Series of the seasonless time series',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'yaxis': {
            'title':"Residus of the seasonal estimation (°C)"
        },
        
        'width': 1200,
        'height': 500,
        'legend':{
            'title':""
                         }
    })
   
    st.plotly_chart(fig)
    
    st.markdown(r"It is now intersting to test the independence of the elements of the new time series $\{\mathbf{\tilde{M}}_{t,m_t}\}$. But,as in the annual analysis we will use the acf to test the independence of these elements, thus in order to the compute the acf, we must have that the time series is stationary. And this is indeed the case, since we obtain a p-value of $p = 4.231e-05$ to Augmented Dickey-Fuller test, and thus we can reject the null hypothesis according to which the time serie is non-stationnary. Thus, we can look at the plot of the acf and the plot of the pacf, in order to obtain some information about the correlation structure in the time series.")
    
    st.image("/Users/kieranvaudaux/Documents/SCV/SCV_project2.0/notebooks/figure/Monthly/Seasonless_mean_acf_pacf.png")
    
    with st.expander("Explanation of the test for the acf and pacf plot"):
        st.markdown(r'On the two plot above, the blue zone corresponds to an approximate confidence interval for the acf and pacf under the null hypothesis that the acf and pacf are computedf from a iid sequence. This test is based on the fact that for large n the sample autocorrelations of an iid sequence $Y_1, . . . , Y_n$ with finite variance are approximately iid with distribution $\mathbb{N}(0, \frac{1}{n})$. We can therefore test whether or not the observed residuals are consistent with iid noise by examining the sample autocorrelations and partial autocorrelations of the residuals and rejecting the iid noise hypothesis if more than three or four out of 50 fall outside the bounds $\pm \frac{1.96}{\sqrt{n}}$ or if one falls far outside the bounds.')
    
    st.markdown(r"On these two plots, we see that there is a lot of points outside the blue zone, which is a indication to the presence of dependency between the elements of the time series. In order to confirm this suspicion of dependence between elements, we have on the following figure the p-values of the Ljung-Box and McLeod-Li tests for different sizes of auto-correlation sequences.")
    
    pval_ind1 = pd.read_csv("DataGenerated/Monthly/Monthly1_pvalue_indep_GLS")
    
    fig = make_subplots(
        rows=2, cols=3,
        row_heights=[1,1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None],[None,{"type": "Scatter"},None]],
        x_title = 'Autocorrelation lags'
        )
        
    fig.add_trace(go.Scatter(x=pval_ind1.index+1, y=pval_ind1.Ljung,
        mode = 'lines+markers', name = 'Ljung-Box'),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind1.index+1, y=pval_ind1.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li'),row =2, col = 2)
    fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
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
    fig.update_yaxes(title_text="p-value", col=2)
    
    st.plotly_chart(fig)
    
    with st.expander("Explanation of the Ljung-Box and McLeod-Li tests"):
        st.markdown(r'''
        The idea behind these two tests is, instead of checking to see whether each sample autocorrelation $\rho (j)$ falls inside the bounds $\pm \frac{1.96}{\sqrt{n}}$, it is also possible to consider the either the statistic $\mathbf{Q}_{LB} = n(n+2)\sum_{j=1}^{h} \frac{\rho^2 (j)}{n-j}$ for the Ljung-Boy tests or the statistic $\mathbf{Q}_{ML} = n(n+2)\sum_{j=1}^{h} \frac{\rho_{ww}^2 (j)}{n-j}$ for the McLeod-Li test, where $\rho (j)$ is the acf and $\rho_{ww}(j)$ if the acf the squared data. Then, under the null hypothesis that our data are iid, this both statistics could be approximated by a chi-squared distribution with $h$ degree of freedom.
        ''')
        
    st.markdown(r"Although many of the p-values in the McLeod-Li test do not allow us to reject the hypothesis of independence, we have that the p-values in the Ljung-Box test are all well below the significance level of $\alpha = 0.05$, so they allow us to reject the null hypothesis of independence. Then, we now have choose a appropriate time series model to model our time series $\{\mathbf{\tilde{M}}_{t,m_t}\}$. The following figure allows us to compare several models for the time series thanks to the Akaike information criterion (AIC). We use the AIC in order to make a model selection since it is an estimator of prediction error and thereby relative quality of statistical models for our set of data. We have tried to model the time series as arising from an $ARMA(p,q)$ for $p\in\{0,...,4\}$ and $q\in\{0,...,11\}$. Our choice of restricting $p$ and $q$ is mainly due to the fact that we wanted to try to keep the model as simple as possible to model our data. To estimate the coefficients of a given model, we use the method 'ARIMA' in the \textbf{statsmodels} packages on python.")
    
    aic1 = pd.read_csv("DataGenerated/Monthly1_modelSelection.csv")
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None]],
        x_title = 'AR(p) parameters'
        )
    p = np.arange(0,12)
    for i in range(5):
        fig.add_trace(go.Scatter(x=p, y=aic1[aic1.param_q==i].AIC,
            mode = 'lines+markers', name = 'q = '+str(i),
            marker=dict(size=12,color=px.colors.qualitative.Dark24[i])),row =1, col = 2)
    
    #fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'Time series model selection for the seasonless mean temperature',
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

    st.markdown(r'By choosing the model which minimises the AIC, we are led to consider the model $ARMA(4,3)$ to model the time series $\{\mathbf{\tilde{M}}_{t,m_t}\}$. On the following plot, we see the residus of the estimation of the trendless time series as a $ARMA(4,3)$ process, where the "ARIMA" method estimate the coefficient with $\hat{\phi}_1 = -0.039238, \hat{\phi}_2 = 0.310470 , \hat{\phi}_3 = 0.893256, \hat{\phi}_4 = -0.180764, \hat{\theta}_1 = 0.225011, \hat{\theta}_2 = -0.210028, \hat{\theta}_3 = -0.917060 \text{ and } \hat{\sigma}^2 = 2.332686$ where the $\hat{\phi}_i$ are the autoregressive coefficients, the $\hat{\theta}_i$ are the moving average coefficients.')
    
    resid_arma4_3 =pd.read_csv("DataGenerated/Monthly/Monthly_arma4_3.csv")
        
    fig = px.line(resid_arma4_3, x='Date', y='resid')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6 months", step="month", stepmode="backward"),
                dict(count=1, label="1 year", step="year", stepmode="backward"),
                dict(count=10, label="10 years", step="year", stepmode="backward"),
                dict(count=30, label="30 years", step="year", stepmode="backward"),
                dict(count=50, label="50 years", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    fig['layout'].update({
        'title': 'Residus of the ARMA(4,3) model',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'yaxis': {
            'title':"Residus mean temperature(°C)"
        },
        
        'width': 1200,
        'height': 500,
        'legend':{
            'title':""
                         }
    })
   
    st.plotly_chart(fig)
    
    st.markdown(r'In order to confirm the consistency of our model, we tested the independence and the distribution of the residuals obtained. The following figure show us the result of the Ljung-Box and McLeod-Li test on the residuals of our ARMA(4,3) model, we see that all p-values are far above the significance threshold of $\alpha = 0.05$, and thus do not allow us to reject the null hypothesis of independence of the tests. Therefore, the $ARMA(4,3)$ model seems to capture well the correlation structure between the elements of the time series $\{\mathbf{\tilde{M}}_{t,m_t}\}$.')
    
    
    pval_ind2 = pd.read_csv("DataGenerated/Monthly/Monthly2_pvalue_indep_GLS")
    
    fig = make_subplots(
        rows=1, cols=3,
        row_heights=[1],
        column_widths=[1./10,8./10,1./10],
        specs=[[None,{"type": "Scatter"},None]],
        x_title = 'Autocorrelation lags'
        )
        
    fig.add_trace(go.Scatter(x=pval_ind2.index+1, y=pval_ind2.Ljung,
        mode = 'lines+markers', name = 'Ljung-Box'),row =1, col = 2)
    fig.add_trace(go.Scatter(x=pval_ind2.index+1, y=pval_ind2.McLeod,
        mode = 'lines+markers',name = 'McLeod-Li'),row =1, col = 2)
    #fig.add_hline(y=0.05,line_dash="dash", line_color="red", name = 'alpha = 0.05')
    
    fig['layout'].update({
        'title': 'P-values of the Ljung-Box and McLeod-Li tests on the residus of the ARMA(4,3)',
        'title_x': 0.5,
        'xaxis': {
            'zeroline': False
        },
        'yaxis': {
        },
        'width': 1200,
        'height': 500,
    })
    fig.update_yaxes(title_text="p-value")
    
    st.plotly_chart(fig)
    
    st.markdown(r"Moreover,as we could expect the residues are of the ARMA(4,3) model are stationary since we obtain a p-value of $p = 0$ for the Augmented Dickey-Fuller test. Thus, we also could check the plots of the acf anf pacf, on the followings figures, to asses the remaining correlation structure in the residus. And we could see that all the autocorrelation lies in the blue 95% confidence interval, which confirms that it is reasonable to assume that the residuals are independent. ")
    
    st.image("/Users/kieranvaudaux/Documents/SCV/SCV_project2.0/notebooks/figure/Monthly/Seasonless_mean_acf_pacf.png")
    
    st.markdown(r'Nevertheless, after performing the Jarque-Bera test, which has the null hypothesis that our data are from a normal distribution, and the Goldfeld-Quandt test, which has the null hypothesis that the data are homoskedastic, we get the p-values : ')

    col1,col2,col3 = st.columns([1,1,1])
    
    with col2:
        st.latex(r'''\text{Normality test : }p_{value} = 8.411e-15 \\
                \text{Heteroskedasticity test : }p_{value} = 0.787
        ''')
        
    with st.expander("Jarque-Bera and Goldfeld_Quandt tests"):
        st.markdown(r'''$\textbf{The Jarques-Bera test}$ :  is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution. The test statistic is defined as : $JB = \frac{n}{6} \left( S^2 + \frac{1}{4} \left(K-3\right)^2 \right)$, where $S =\frac{ \frac{1}{n} \sum_{i=1}^{n} (x_i-\bar{x})^3 }{ \left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^{\frac{3}{2}} }$ and $K =\frac{ \frac{1}{n} \sum_{i=1}^{n} (x_i-\bar{x})^4 }{ \left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^{2} }$.\\
            If the data comes from a normal distribution, the JB statistic asymptotically has a chi-squared distribution with two degrees of freedom, so the statistic can be used to test the hypothesis that the data are from a normal distribution. The null hypothesis is a joint hypothesis of the skewness being zero and the excess kurtosis being zero.''')
        st.markdown(r'''$\textbf{The Goldfeld-Quandt tests}$ : is accomplished by undertaking separate least squares analyses on two subsets of the original dataset: these subsets are specified so that the observations for which the pre-identified explanatory variable takes the lowest values are in one subset, with higher values in the other. The subsets needs not be of equal size, nor contain all the observations between them. The test assumes that the errors have a normal distribution. There is an additional assumption here, that the design matrices for the two subsets of data are both of full rank. The test statistic used is the ratio of the mean square residual errors for the regressions on the two subsets. This test statistic corresponds to an F-test of equality of variances.
            ''')
    
    st.markdown(r"Even if the p-value of the test is significant at a threshold of $\alpha = 0.05$, we have that the normality test is widely rejected. Thus, this leads us to believe that the model we have developed so far may not be appropriate for our data. ")
