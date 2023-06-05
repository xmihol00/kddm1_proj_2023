# Model Training and Evaluation
The following sections summarize how we trained and evaluated our machine learning models. We reference the columns of the University dataset by their indices, mapping from column names is available in the following table:
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
We used 80/20 dataset split to train and evaluate all models. Additionally, we implemented cross validation to tune hyper-parameters of our models since the dataset is very small. We fixed the randomness for the dataset split, weight initialization, cross validation fold selection as well as the batch selection during training using a random seed of 14 (our group number). We evaluated our models with the mean squared error, root mean squared error, mean absolute error and R2 score metrics. 

## Models
We describe how we trained and found the best hyper-parameters for our models in the following sections.

### Baseline - Linear Regression
We trained the simples possible model, i.e. linear regression model, on all the continuous and categorical columns column to have a baseline for comparing our more advanced models. No hyper-parameter tuning was necessary for this model.

The performance of this model is summarized in the following table:
| Training Column Indices                               |         MSE |    RMSE |     MAE | R2 score |
|-------------------------------------------------------|------------:|--------:|--------:|---------:|
| 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 |  2647955.42 | 1595.23 | 1172.27 |   0.6722 |

### Random Forest
In Random Forest Regression, each tree in the ensemble is built from a sample drawn with replacement from the training set.
It combines multiple decision trees to make predictions.

Advantages of Random Forest Regression:

- Perform on large data sets with high dimensionality
- Less prone to overfitting, than single decision trees.
- Capture complex nonlinear relationships.
- Estimates of feature importance.

Disadvantages of Random Forest Regression:

- Bad on noisy or irrelevant features.
- Computationally expensive for complex datasets.

We applied `GridSearchCV` to find the best parameter for the number of estimators by the negative mean squared error.
The best parameters found at single parameter search are `n_estimators=29` and `max_features=8`
For combine parameter search `n_estimators=39` and `max_features=17` yield the best performance.
We also performed some empirical approaches.
The overall best result was gained by `n_estimators=100` and `max_features=10`.
Thus we trained with this parameters. The returned accuracy is 0.780.

| Model                    | Training Column Indices                            |        MSE |    RMSE |     MAE | R2 score |
| -------------------------|----------------------------------------------------|-----------:|--------:|--------:|---------:|
| Random Forest            | 6, 5, 7, 8, 11, 18, 14, 16, 12, 13                 | 1771101.19 | 1297.59 |  981.13 |   0.783  |

### Support Vector Machine

### Neural Network
First, we defined 3 neural network architectures listed below:
    - non-linear with 2 hidden layers, first with a ReLU activation and second with a linear activation,
    - non-linear with 3 hidden layers, first two with a ReLU activation and last with a linear activation,
    - non-linear with 4 hidden layers, first three with a ReLU activation and last with a linear activation,
The number of neurons in the input layer varies based on the selected subset of used columns for training, see below. The numbers of neurons in the following hidden layers is the same as in the input layer. The output layer has 2 neurons given the two predicted variables.

Second, we trained the defined models using a 3-fold cross validation scheme with an early stopping with patience for 25 epochs as the stopping criterion. The batch size was set to 8, since the dataset is very small. Several optimizers with different learning rate configurations were tested, the best performing was the Adam optimizer with quite high learning rate of 0.05. Each of the models was trained on all the training datasets, i.e. with missing values imputed with the mean, the median and with mixed missing value imputation using various techniques. Additionally, we selected 3 subsets of the available columns in the training dataset to repeatably train on. The subsets are the following:
    - all continuous and categorical columns,
    - only continuous columns excluding the columns `Latitude` and `Longitude`,
    - columns with absolute value of correlation higher than 0.5 (`UK_rank`, `World_rank`, `CWUR_score`, `Minimum_IELTS_score`, `International_students`, `Academic_staff_from`, `Academic_staff_to`) with the target variables.

Third, we re-trained the on average best performing model on the cross validation folds. The selected model was trained using the whole training dataset with the on average best performing subset of columns and the average number of epochs recorder during the cross validation runs as the stopping criterion.

The best performing model was the non-linear model with 2 hidden layers trained using the selected columns from the mixed missing value imputed dataset. The performance of this model is summarized in the following table:
| Training Column Indices |        MSE |    RMSE |    MAE | R2 score |
|-------------------------|-----------:|--------:|-------:|---------:|
| 5, 6, 7, 8, 11, 14      | 1789915.76 | 1309.89 | 983.00 |   0.7789 |

## Ensemble
We decided to further improve the prediction by creating an ensemble simply by averaging the predictions outputs of our models. The performance of the ensemble is summarized in the following table:
|        MSE |    RMSE |     MAE | R2 score |
|-----------:|--------:|--------:|---------:|
|            |         |         |          |

## Performance Comparison Table
| Model                    | Training Column Indices                               |        MSE |    RMSE |     MAE | R2 score |
| -------------------------|-------------------------------------------------------|-----------:|--------:|--------:|---------:|
| baseline - linear        | 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 | 3654911.02 | 1886.08 | 1384.17 |   0.5416 |
| Neural Network           | 5, 6, 7, 8, 11, 14                                    | 1789915.76 | 1309.89 |  983.00 |   0.7789 |
| Random Forest            | 6, 5, 7, 8, 11, 18, 14, 16, 12, 13                    | 1771101.19 | 1297.59 |  981.13 |   0.783  |
| Support Vector Machine   |                                                       |            |         |         |          |
| Ensemble                 |                                                       |            |         |         |          |

## Summary
