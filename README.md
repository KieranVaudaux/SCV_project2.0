
# Statistical Data Analysis on Meteorological Data

This project is done in the context of the course *Statistical Computation and Visualization*, taught by [Mehdi Gholam](https://people.epfl.ch/mehdi.gholam?lang=fr) at EPFL.

## Tools and Languages

<img align="left" alt="Visual Studio Code" width="26px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/visual-studio-code/visual-studio-code.png" />

<img align="left" alt="Python" width="26px" src="https://camo.githubusercontent.com/0fd2667849df9f18b863a2fc9fdf275d28c0e69bae657009213dbbba08295d02/68747470733a2f2f7261772e6769746875622e636f6d2f436972636c6543492d5075626c69632f63696d672d707974686f6e2f6d61737465722f696d672f636972636c652d707974686f6e2e7376673f73616e6974697a653d74727565" />

<img align="left" alt="C++" width="26px" 
src="https://raw.githubusercontent.com/isocpp/logos/master/cpp_logo.png" />

<img align="left" alt="Overleaf" width="26px" 
src="https://pbs.twimg.com/profile_images/551035690234834945/JhdUiOPP.png" />

<br />
<br />

## Table of Content

* [People](#people)
* [Description](#description)
* [Aims](#aims)
* [Streamlit Web App](#streamlit)
* [Project Organization](#project-organization)
* [Related Articles and Useful References](#refs)
* [Interesting Material 🔍](#material)

## People

* [Luca Bracone](https://people.epfl.ch/luca.bracone) ([GitHub](https://github.com/jkasalt)) 
* [Luca Nyckees](https://people.epfl.ch/luca.nyckees) ([GitHub](https://github.com/LucaNyckees)) 
* [Blerton Rashiti](https://people.epfl.ch/blerton.rashiti) ([GitHub](https://github.com/BlertonRashiti)) 
* [Kieran Vaudaux](https://people.epfl.ch/kieran.vaudaux) ([GitHub](https://github.com/KieranVaudaux)) 

## Description

We are interested in the study of meteorological data from the Geneva Observatory in Switzerland. More specifically, we are interested in the temporal evolution of the average temperature from 1901 to now. We aim to model the evolution of the mean temperature, in order to see if we can observe a significant increasing trend in it. In particular, we use various Python visualisation tools to allow an intuitive interactive framework. The dataset that we use for data analysis can be found [here](https://www.ecad.eu/utils/showselection.php?99j9a2jpggb49ha5t4mc9evpol).

<img width="450" alt="figure" src="https://github.com/LucaNyckees/SCV_project1/blob/main/figures/temperatures_image.png">

## Aims

Within the statistical data analysis we make, we aim at answering a set of specific questions :

* Can we make predictions on certain meteorological features, such as mean temperature ?
* Can we establish a link (correlation and causality) between various meteorological features, such as precipitation and mean temperature ?
* Can we predict the behavior of a single meteorological feature based on the data of several other correlated features ?

## Streamlit Web App

You can launch the Streamlit web application with the following commands. First, open a shell/terminal and go to the directory in which you saved the project - for example :

```
cd Desktop/levelset/zigzag
```
Then, go directly to the source code with 

```
cd src
```

Finally, type the command below in your shell and enjoy the app!
```
streamlit run st_app.py
```
For an original theme configuration, you may replace the last command with this one :
```
streamlit run st_app.py --theme.primaryColor="#3271e2" --theme.backgroundColor="#357dc5" --theme.secondaryBackgroundColor="#68708c" --theme.textColor="#dadde6"
```

## Project Organization
------------

    ├── README.md          -- Top-level README.
    │
    ├── notebooks          -- Jupyter notebooks.
    │
    ├── articles           -- Related articles and useful references.
    │
    ├── reports            -- Notes and report (Latex, pdf).
    │ 
    ├── figures            -- Optional graphics and figures to be included in the report.
    │
    ├── data               -- Raw data.
    │
    ├── requirements.txt   -- Requirements file for reproducibility.
    └── src                -- Project source code.
   
--------

## Related Articles and Useful References

[[1]](https://arxiv.org/abs/1902.06183) - A Statistical Analysis of Noisy Crowdsourced Weather Data\
[[2]](https://arxiv.org/pdf/2103.10936.pdf) - Forecasting of Meteorological
variables using statistical methods
and tools

## Interesting Material 🔍

+ General overview on meteorological data analysis [[click here]](https://www.sciencedirect.com/topics/social-sciences/meteorological-data)
+ Video lectures on meteorological data visualization [[click here]](https://www.youtube.com/watch?v=E_n3Ft4WozM)

