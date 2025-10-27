# Data Challenge: Capgemini x Mines Paris PSL 2025

This repository contains the 1st place solution for the data challenge organized by Capgemini and Mines Paris PSL. The competition ran for two weeks. All instructions can be found at https://www.kaggle.com/competitions/hackathon-mines-paris-2025.

### The Challenge

The task was to predict water flow at various monitoring stations across France and Brazil for three different time horizons: one week, two weeks, and three weeks in advance.

The data provided included:
* **Time Series Data:** Station-specific measurements like evaporation and precipitation.
* **Static Data:** Geographical and soil composition data for each station.

### The Solution

The core of the solution is a Boosted Tree Regressor trained on a rich set of handcrafted features with cautious regularization. The final prediction is an average (bagging) of five separate models trained with different random seeds.

### Key Methodological Decisions
* **Boosted Tree Regressor:** Suited when many features. Out of shelf, explanable, well optimized, lightweight.
* **Separate Models:** We trained one independent model for each prediction horizon (+1 week, +2 weeks, +3 weeks).
* **Country-Specific Models:** During exploration, we found that a single global model performed poorly because France and Brazil have opposite seasonal patterns. The final solution trains **separate models for each country**, which proved much more effective.
* **Feature Engineering:** This was critical. We created many non-linear features, including:
    * **Seasonal Features:**: sine functions for day, week, and month of the year.
    * **Interaction Features:** Combining weather and soil data (e.g., `precipitations` $\times$ `soil_moisture_lag_1` or `precipitations` $\times$ `clay`).
    * **Lagged Features:** Using past values of time series like `temperatures` and `soil_moisture` from previous weeks.
    * **Rolling Features:** Moving averages and standard deviations to capture trends.
* **Model Selection:** We experimented with both LightGBM and XGBoost. LightGBM showed a slight advantage, though they weren't in a perfectly identical setup.
* **Regularization:** Focusing on limiting tree structure (depth, leaves), forced random splits, and regularization (L1/L2). Used model-based feature selector, also boosted tree regressor to drop 50% of features.
*  **Robustness to outlier:** Overall data was clean, but we choosed the huber loss as objective function instead of the mean squared.
* **Bagging:** Since the final regressor was lightweight. We trained 5 instances with the same optimal hyperparameters but different random seeds. The final submission is the average of these 5 models, which makes the predictions more stable.

### Validation Strategy

To ensure the model could generalize before submitting to Kaggle, we evaluated it on two hold-out splits *before* submission, created in the training data as follows:

1.  **Temporal Split:** Stations present in the training set, but at more recent dates.
2.  **Spatio-Temporal Split:** Stations *not* present in the training set, also at more recent dates.

### Limitations and Further Steps:
- Training a model to predict the 3 time horizons at the time, or 3 models but the predictor at +1week uses prediction of the +0 week predictor