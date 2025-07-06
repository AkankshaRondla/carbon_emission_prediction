# Carbon Emission Prediction and Forecasting

This repository contains Jupyter notebooks for the analysis, prediction, and forecasting of CO₂ emissions. The project involves data cleaning, exploratory data analysis (EDA), machine learning model training, and forecasting of future CO₂ emissions per capita.

## Table of Contents

  * [Project Overview](#project-overview)
  * [Files Description](#files-description)
  * [Setup and Installation](#setup-and-installation)
  * [Usage](#usage)
  * [Results and Visualizations](#results-and-visualizations)
  * [Contributing](#contributing)

## Project Overview

This project aims to predict and forecast CO₂ emissions using various environmental and economic indicators. It leverages machine learning techniques, specifically `RandomForestRegressor`, to build a predictive model and then uses this model to forecast CO₂ emissions per capita for future years.

The workflow is divided into three main parts:

1.  **Data Preparation**: Cleaning and preprocessing the raw dataset.
2.  **Exploratory Data Analysis (EDA)**: Understanding the data, its distributions, and relationships between variables.
3.  **Model Training and Forecasting**: Developing a machine learning model, evaluating its performance, and generating future forecasts.

## Files Description

  * `Carbon_Emision_Prediction[1].ipynb`: This notebook handles the initial data cleaning and preprocessing steps. It identifies and removes rows with any missing values and ensures data quality. The cleaned data is then exported to a CSV file named `data_cleaned.csv` for use in subsequent notebooks.
  * `Carbon_Emision_Prediction2 (1).ipynb`: This notebook focuses on exploratory data analysis (EDA) and data understanding. It loads the `data_cleaned.csv` file, provides an overview of the dataset's shape and data types, and generates descriptive statistics. Key visualizations include a "4D plot" which explores the relationships between `urb_pop_growth_perc`, `co2_per_cap`, `en_per_cap`, and `pop_urb_aggl_perc`, providing insights into how urban population dynamics, energy consumption, and urbanization levels interact with CO₂ emissions.
  * `Carbon_Emision_Prediction3.ipynb`: This notebook contains the core machine learning pipeline. It includes data splitting using `train_test_split`, model training with `RandomForestRegressor`, and performance evaluation using cross-validation (`cross_val_score`) and metrics such as `r2_score` and `mean_squared_error`. The notebook generates forecasts for CO₂ emissions per capita for the next 20 years, visualizes these forecasts, and saves the trained model as `model.pkl`. It also specifically prints the forecasted CO₂ values for India for the last 5 years in the forecast period.

## Setup and Installation

To run these notebooks, you will need to have Python installed along with the following libraries:

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn` (`sklearn`)
  * `statsmodels`
  * `joblib`

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels joblib
```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AkankshaRondla/carbon_emission_prediction/
    cd carbon_emission_prediction
    ```
2.  **Run the notebooks in sequence:**
      * Start with `Carbon_Emision_Prediction[1].ipynb` to clean the data and generate `data_cleaned.csv`.
      * Proceed to `Carbon_Emision_Prediction2 (1).ipynb` for EDA and data visualization.
      * Finally, run `Carbon_Emision_Prediction3.ipynb` to train the model, evaluate it, and generate CO₂ emission forecasts.

## Results and Visualizations

The `Carbon_Emision_Prediction3.ipynb` notebook includes a plot titled 'Forecasted CO₂ Emissions per Capita (Next 20 Years)', which visualizes the future trends of CO₂ emissions based on the trained model. Additionally, `Carbon_Emision_Prediction2 (1).ipynb` contains various plots for understanding the dataset characteristics and relationships between features, including a "4D plot" that helps in visualizing complex interactions between multiple variables.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests.
