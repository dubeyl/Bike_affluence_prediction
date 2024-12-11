# Social Media Analytics - Bike sharing analysis

This project investigates the prediction of bike-sharing demand in Montreal using data from Bixi, a bike-sharing company. Data exploration revealed significant patterns, including a substantial increase in bike usage over the past decade, seasonal trends with higher demand in warmer months, and distinct weekday peaks due to commuter activity. The study compares two predictive models: a Geographically Weighted Regression (GWR) and a multi-linear regression (MLR). While GWR incorporates spatial variability, the simpler MLR model demonstrates superior runtime efficiency and accuracy. The findings indicate that the MLR model is more effective for predicting bike-sharing demand, aiding in better service coverage and balance between bike availability and demand.

The current GitHub repository contains the script used to generate all of the information provided in the project report.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Installation

Detailed instructions on how to install the project.

### Prerequisites

For a smooth usage of the script, please the latest version of python installed and the required packages (see requirements.txt)

## Usage

We don't recommend to start any of the script but enjoy the provided datasets and visualisations.
However, if you follow the instruction in each script, you should be able to run them smoothly.

Your project should have the following structure:
* ./bixi
    * Code
        * 01_data_cleaning.py
        * 02_data_transform2neo4j.ipynb
        * 02_neo4j-admin_import.txt
        * 03_add_weather_distance.ipynb
        * 03_analysis.ipynb
        * 04_month_analysis.ipynb
        * 05_predict_compare_models_MLR_GWR.ipynb
        * 05_predict_working_GWR.ipynb
        * 09_live_demo.ipynb
        * fct_helper.py
        * README
        * requirements.txt
    * Data
        * 2014.csv
        * 2015.csv
        * 2016.csv
        * 2017.csv
        * 2018.csv
        * 2019.csv
        * 2020.csv
        * 2021.csv
        * 2022.csv
        * 2023.csv
        * 2023_edges.csv
        * 2023_edges_extended.csv
        * 2023_edges_extended_header.csv
        * 2023_edges_header.csv
        * 2023_forecast_hourly.csv
        * 2023_nodes.csv
        * 2023_nodes_header.csv
        * 2024_0102.csv
        * distances_df.csv
        * original
            * 2014_stations.csv
            * 2014-04_opendata.csv
            * 2014-05_opendata.csv
            ... (lot of files)
            * 2021-00_opendata.csv
            * 2022.csv
            * 2023.csv
            * 2024_0102.csv


You can download the required dataset at one of the following location:
* https://gofile.me/4lFtV/Uxl0PJJ0k (with original, available until the end of August 2024)
* [Kaggle SMA Bixi](https://kaggle.com/datasets/1064fa190a636f6dbd2020dfa6de620acfe1754299e56cdec865ee1d08a27994) (without original)
* Original data only: https://bixi.com/en/open-data/

!!! The datasets are large. You might want to focus only on some of them according to the scripts you want to use.

## Contact

If you may have difficulties running the provided script or any other question, you can contact the authors at:
* leandre.dubey@gmail.com
* louis.chaubert@students.unibe.ch

