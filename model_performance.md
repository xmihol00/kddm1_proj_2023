# Model Performance
The following sections summarize the performance of our machine learning models. The columns referenced by their indices, mapping from column names is available in the following table:
|  Index  | Column                                         |
|---------|------------------------------------------------|
|  0      | Id                                             |
|  1      | University_name                                |
|  2      | Region                                         |
|  3      | Founded_year                                   |
|  4      | Motto                                          |
|  5      | UK_rank                                        |
|  6      | World_rank                                     |
|  7      | CWUR_score                                     |
|  8      | Minimum_IELTS_score                            |
|  9      | UG_average_fees_(in_pounds)                    |
|  10     | PG_average_fees_(in_pounds)                    |
|  11     | International_students                         |
|  12     | Student_satisfaction                           |
|  13     | Student_enrollment                             |
|  14     | Academic_staff                                 |
|  15     | Control_type                                   |
|  16     | Academic_Calender                              |
|  17     | Campus_setting                                 |
|  18     | Estimated_cost_of_living_per_year_(in_pounds)  |
|  19     | Latitude                                       |
|  20     | Longitude                                      |
|  21     | Website                                        |

## Common Practice
We used 80/20 dataset split to train and evaluate all models. Additionally, we used cross validation to tune hyper-parameters of our models since the dataset is very small.

## Baseline - Linear Regression
We implemented the simples possible model, i.e. linear regression model, on all the available columns to have a baseline for comparing our more advanced models. 

The performance of this model is summarized in the following table:
| MSE        | RMSE       | MAE        |
|            |            |            |

## Neural Network

## Random Forest

## Support Vector Machine

## Performance Comparison Table
| Student            | Model                    | Training Column Indices         | MSE        | RMSE       | MAE        | Accuracy |
|--------------------| -------------------------|---------------------------------|------------|------------|------------|----------|
| Alexander          | x                        | X                               | x          |            |            | x        |
| David              | NN                       | 5, 6, 7, 8, 11, 12, 13, 14, 18  | 2971490.75 |            |            | x        |
| Ronald             | random forest            | 10                              | 3074441.13 |            |            | 0.49     |
| Thomas             | support vector machine   | X                               | x          |            |            | x        |

## Ensemble
