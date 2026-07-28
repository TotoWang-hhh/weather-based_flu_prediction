# Disease Prediction Method Summary

This document is based on the current implementation in the project and the latest experimental outputs. It explains each prediction approach in plain English and summarizes their strengths, limitations, and current performance.

## Research Objective

The goal of this project is to predict weekly new disease cases using weather data and historical case data. The core idea is to combine weather features, lagged features, and historical case trends, then compare different modeling strategies to see which one performs best.

## Description of Each Approach

| Method                  | Plain-English Description                                                                                                                                                                                                                                         | Current Performance                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| Linear Regression       | This is the simplest baseline model. It assumes that the relationship between weather features and new cases can be represented by a straight-line pattern. It is easy to implement and interpret, but it usually struggles with nonlinear and seasonal patterns. | MAE 4033.69, RMSE 4779.09, R² -0.4049 |
| Random Forest           | This is a nonlinear machine-learning model based on many decision trees. It can capture more complex feature interactions than linear regression, but it still has limited ability to model temporal dependence directly.                                         | MAE 3376.79, RMSE 3863.12, R² 0.0820  |
| Standard LSTM           | This is a sequence model that learns patterns from a series of past observations. It is designed to capture temporal dynamics, which is important when predicting disease trends over time.                                                                       | MAE 2754.83, RMSE 3599.96, R² 0.2508  |
| History-Weather LSTM    | This version uses both historical weather information and prior case information as input. It is intended to better reflect the delayed effect of weather conditions on disease transmission over several weeks.                                                  | MAE 2861.64, RMSE 3314.88, R² 0.3241  |
| Notebook-Inspired LSTM  | This approach follows the structure of a reference notebook-based experiment. It uses a deeper and more complex sequence architecture in order to capture richer nonlinear patterns.                                                                              | MAE 3209.02, RMSE 3688.02, R² 0.1634  |
| Stacking Meta-Model     | This is a two-level fusion approach. Several base models first produce predictions, and then a second-stage model learns how to combine those predictions into a final result. The idea is to let different models contribute different strengths.                | MAE 3866.90, RMSE 4313.37, R² -0.1444 |
| Weighted Ensemble Model | This is a simpler fusion method. Predictions from multiple models are combined using weighted averaging based on their individual performance. It is similar to asking several experts to vote and then combining their opinions.                                 | MAE 2761.51, RMSE 3089.94, R² 0.4414  |

## Overall Conclusions

1. Linear regression performs the worst in this task, which suggests that the relationship between weather and disease cases is not simple or linear.
2. LSTM-based models generally outperform traditional machine-learning models, indicating that temporal information is valuable for this prediction problem.
3. Fusion-based methods show clear benefits over single models in this setting, especially the weighted ensemble model.
4. Based on the current results, the weighted ensemble is the most competitive approach among the tested methods.

## Notes

- The main implementation is in main.py.
- Output files are stored in the outputs directory, including model metrics, prediction comparison tables, and charts.
- Future improvements could include better feature engineering, a longer historical window, or a stronger second-stage fusion model.
