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
We trained the simples possible model, i.e. linear regression model, on all the continuous and categorical columns excluding the `Founded_year` column to have a baseline for comparing our more advanced models. No hyper-parameter tuning was necessary for this model.

The performance of this model is summarized in the following table:
| Training Column Indices                            |         MSE |    RMSE |     MAE | R2 score |
|----------------------------------------------------|------------:|--------:|--------:|---------:|
| 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 |  3654911.02 | 1886.08 | 1384.17 |   0.5416 |

### Random Forest

### Support Vector Machine

### Neural Network
First, we defined 4 neural network architectures listed below:
    - linear with 1 hidden layer with linear activation,
    - non-linear with 2 hidden layers, first with a ReLU activation and second with a linear activation,
    - non-linear with 3 hidden layers, first two with a ReLU activation and last with a linear activation,
    - non-linear with 4 hidden layers, first three with a ReLU activation and last with a linear activation,
The number of neurons in the input layer varies based on the selected subset of used columns for training, see below. The numbers of neurons in the following hidden layers is the same as in the input layer. The output layer has 2 neurons given the two predicted variables.

Second, we trained the defined models using a 3-fold cross validation scheme with an early stopping with patience for 25 epochs as the stopping criterion. The batch size was set to 8, since the dataset is very small. Several optimizers with different learning rate configurations were tested, the best performing was the Adam optimizer with quite high learning rate of 0.05. Each of the models was trained on both the training datasets with missing values imputed with the mean and the median. Additionally, we selected 3 subsets of the available columns in the training dataset to repeatably train. The subsets are the following:
    - all continuous and categorical columns,
    - only continuous columns excluding the columns `Latitude` and `Longitude`,
    - columns with absolute value of correlation higher than 0.5 (`UK_rank`, `World_rank`, `CWUR_score`, `Minimum_IELTS_score`, `International_students`, `Academic_staff_from`, `Academic_staff_to`).

Third, we trained the on average best performing model on the cross validation folds, which reached its best average performance after more than 10 epochs on average. The requirement was employed to ensure that the model does not start overfitting the data too soon. Subsequently, the selected model was trained using the whole training dataset with the on average best performing subset of columns and the average number of epochs recorder during the cross validation runs as the stopping criterion.

The best performing model was the non-linear model with 3 hidden layers trained using continuous columns excluding the columns `Latitude` and `Longitude` from the dataset with missing values imputed with the median. The performance of this model is summarized in the following table:
| Training Column Indices        |        MSE |    RMSE |     MAE | R2 score |
|--------------------------------|-----------:|--------:|--------:|---------:|
| 5, 6, 7, 8, 11, 12, 13, 14, 18 | 2739160.15 | 1652.33 | 1264.20 |   0.6422 |

## Ensemble
We decided to further improve the prediction by creating an ensemble simply by averaging the predictions outputs of our models. The performance of the ensemble is summarized in the following table:
|        MSE |    RMSE |     MAE | R2 score |
|-----------:|--------:|--------:|---------:|
|            |         |         |          |

## Performance Comparison Table
| Model                    | Training Column Indices                            |        MSE |    RMSE |     MAE | R2 score |
| -------------------------|----------------------------------------------------|-----------:|--------:|--------:|---------:|
| baseline - linear        | 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 | 3654911.02 | 1886.08 | 1384.17 |   0.5416 |
| x                        |                                                    |            |         |         |          |
| Neural Network           | 5, 6, 7, 8, 11, 12, 13, 14, 18                     | 2739160.15 | 1652.33 | 1264.20 |   0.6422 |
| Random Forest            |                                                    | 2379330.21 | 1521.14 | 1071.26 |   0.7019 |
| Support Vector Machine   |                                                    |            |         |         |          |
| Ensemble                 |                                                    |            |         |         |          |

## Summary