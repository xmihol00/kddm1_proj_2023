# Model Training and Evaluation
The following sections summarize how we trained and evaluated our machine learning models. We provide a only table summarizing the data set below. More information about the data set and performed pre-processing is available in the [dataset_analysis.md](dataset_analysis.md) file. 

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
We used 80/20 dataset split to train and evaluate all models. Additionally, we implemented 5-fold cross validation to tune hyper-parameters of our models since the dataset is very small. We fixed the randomness for the dataset split, weight initialization, cross validation fold selection as well as the batch selection during training using a random seed 42 for the models and seeds $[40, 49]$ for the data set to get more representative results less affected by the data set spilt. Moreover, we performed the cross validation on 3 differently imputed data sets (mean, median and mixed, see [dataset_analysis.md](dataset_analysis.md)) and on 3 subsets of the available columns:
    - all continuous and categorical columns (*all*),
    - only continuous columns excluding the columns `Latitude` and `Longitude` (*continuous*),
    - columns with absolute value of correlation higher than 0.5 (`UK_rank`, `World_rank`, `CWUR_score`, `Minimum_IELTS_score`, `International_students`, `Academic_staff_from`, `Academic_staff_to`) with the target variables (*selected*).

We evaluated the performance of our models on the cross validation sets and test set with the mean squared error, mean absolute error, root mean squared error and R2 score metrics. Since we run the cross validation as well as the final evaluation on the test set for 10 different seeds, we computed the final results shown in tables below as the median across these runs to eliminate outliers. 

## Models
We describe how we trained and found the best hyper-parameters for our models in the following sections.

### Baseline - Linear Regression
We trained the simples possible model, i.e. linear regression model, to have a baseline for comparison of our more advanced models. No hyper-parameter tuning was necessary for this model.

The performance of the linear regression model is summarized in the following table:

|       MSE |    MAE |   RMSE | R2 score | Best Column Subset | Best Imputation Method |
|----------:|-------:|-------:|---------:|--------------------|------------------------|
| 3708162.9 | 1408.8 | 1909.6 |   0.3075 | continuous         | mixed                  |

### Neural Network
First, we defined 3 neural network architectures listed below:
    - non-linear with 2 hidden layers, first with a ReLU activation and second with a linear activation,
    - non-linear with 3 hidden layers, first two with a ReLU activation and last with a linear activation,
    - non-linear with 4 hidden layers, first three with a ReLU activation and last with a linear activation,
The number of neurons in the input layer varies based on the selected subset of used columns for training, see below. The numbers of neurons in the following hidden layers is the same as in the input layer. The output layer has 2 neurons given the two predicted variables.

Second, we trained the defined models an early stopping with patience for 25 epochs as the stopping criterion. The batch size was set to 8, since the dataset is very small. Several optimizers with different learning rate configurations were tested, the best performing was the Adam optimizer with quite high learning rate of 0.05. 

Third, we re-trained the on average best performing model on the cross validation folds. The selected model was trained using the whole training dataset with the on average best performing subset of columns and the average number of epochs recorder during the cross validation runs as the stopping criterion.

Advantages of neural networks:
- capture complex non-linear relationships,
- flexible, neural networks can handle a wide range of regression problems,
- automatic learning of the relevant features from the input data,
- scalable by increasing the number of hidden layers and neurons.

Disadvantages of neural networks:
- prone to overfitting, especially when dealing with small datasets,
- black box models, their internal workings are not easily interpretable,
- computationally expensive,
- require a substantial amount of labeled training data to perform well.

The best performing model was the model with 4 hidden layers. The performance of this model is summarized in the following table:

|       MSE |    MAE |   RMSE | R2 score | Best Column Subset | Best Imputation Method |  
|----------:|-------:|-------:|---------:|--------------------|------------------------|  
| 3389145.4 | 1304.6 | 1826.1 |   0.3659 | selected           | median                 |  

### Random Forest
In random forest regression, each tree in the ensemble is built from a sample drawn with replacement from the training set.
It combines multiple decision trees to make predictions.

Advantages of random forests:
- perform well on large data sets with high dimensionality,
- less prone to overfitting, than single decision tree,
- capture complex non-linear relationships,
- estimation of feature importance.

Disadvantages of random forests:
- perform poorly on noisy or irrelevant features,
- computationally expensive for complex datasets.

We applied a `GridSearchCV` with `max_features` ranging from 1 to 17 and `n_estimators` ranging from 80 to 100 to find the best hyper-parameters. The best found parameters were `max_features=5` and `n_estimators=98`. The performance of the best random forest model is summarized in the following table:

|       MSE |    MAE |   RMSE | R2 score | Best Column Subset | Best Imputation Method |
|----------:|-------:|-------:|---------:|--------------------|------------------------|
| 3200786.3 | 1192.1 | 1761.7 |   0.4061 | continuous         | median                 |

### Support Vector Regression
The goal of the support vector regression (SVR) is to find a function that approximates the relationship between the input features
and a continuous target, while minimizing the prediction error. It has the same underlying idea as the support vector machine (SVM)
but SVR is focused on regression problems and SVM on classification.

SVR can handle non-linear relationships between features and the target by applying different kernel functions.

Advantages of support vector regression:
- perform well in high-dimensional feature spaces,
- robust to outliers,
- decision model can be easily updated.

Disadvantages of support vector regression:
- not suitable for large datasets,
- perform poorly on noisy data sets or data sets with overlapping target classes.

We applied a `GridSearchCV` with `C` (0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5), `kernel` one of (linear, rbf),  `epsilon` one of (0.0001, 0.0005, 0.005, 0.01, 0.25, 0.5, 1, 5) and `gamma` (0.0001, 0.001, 0.01, 0.1, 1) to find the best hyper-parameters. The best found parameters were `C=5`, `kernel=linear`, `epsilon=5` and `gamma=0.0001`. Unfortunately, even after double checking the implementation prone to overfit on models with a polynomial approach for the kernel:

|       MSE |    MAE |   RMSE | R2 score | Best Column Subset | Best Imputation Method |
|----------:|-------:|-------:|---------:|--------------------|------------------------|
| 3643681.3 | 1219.7 | 1814.7 |   0.4625 | selected           | mixed                  |

## Ensemble
We decided to further improve the prediction performance by creating an ensemble simply by averaging the predictions of the described models above. The ensemble brings a significant improvement in the performance, which is summarized in the following table:

|       MSE |    MAE |   RMSE | R2 score |
|----------:|-------:|-------:|---------:|
| 2363667.7 | 1108.6 | 1537.3 |   0.5178 |

## Performance Comparison Table
| Model                     |       MSE |    MAE |   RMSE | R2 score | Best Column Subset | Best Imputation Method |
| --------------------------|----------:|-------:|-------:|---------:|--------------------|------------------------|
| Linear Regression         | 3708162.9 | 1408.8 | 1909.6 |   0.3075 | continuous         | mixed                  |
| Neural Network            | 3389145.4 | 1304.6 | 1826.1 |   0.3659 | selected           | median                 |
| Random Forest             | 3200786.3 | 1192.1 | 1761.7 |   0.4061 | continuous         | median                 |
| Support Vector Regression | 3643681.3 | 1219.7 | 1814.7 |   0.4625 | selected           | mixed                  |
| Ensemble                  | 2363667.7 | 1108.6 | 1537.3 |   0.5178 |                    |                        | 

## Summary
All the models reached a comparable performance. Additionally, we were able to significantly improve the prediction performance by creating an ensemble of the models.
