# Introduction (slide 1)
Dear audience, welcome to our presentation. My name is X and together with my colleagues Y and Z we will discuss regression. Specifically, we will focus on dataset pre-processing and on prediction with different regression models.

# Outline (slide 2)
The outline of the presentation as well as how we approached the project is the following. First, we researched information about the Universities dataset and performed some analysis. We followed the analysis by cleaning and pre-processing the dataset. Then we defined a training and evaluation scheme used across different models that we tested. Lastly, we analyzed the performance of the chosen models and created an ensemble to see if we cn improve the performance with it.

# Data set Analysis (slides )
TODO

# Cleaning and Pre-processing (slide 5)
Cleaning and pre-processing is the most important part of our project. We used a 6 stage pipeline, which starts with identification of missing values other than NaN. Then, we splitted compounded columns like the academic staff and converted percentage columns to floating point values. Lastly, we removed duplicit entries.

The second stage is very simple, the dataset is splitted to training and test sets 80 to 20. It is important to make the split at this point in order to prevent data leakage from the test set.

We followed up the second stage with missing value imputation replacing missing values using 3 different approaches. First, we used mean imputation of the continuous columns and mode imputation of the categorical columns. Second, we just replaced mean with median and third, we used median, mode, linear regression and K-nearest-neighbors imputation on different columns, as can be seen in the tables. We made sure to impute the missing values only with statistics obtained from the training set.

Then, we normalized all the continuos columns apart form the predicted to zero mean and unit variance. 

Next, we preformed one-hot encoding of the categorical features.

Lastly, we removed all non-numeric columns like the university name.

# Training and Evaluation (slide 6)
We used a 5-fold cross validation to find the best hyper-parameters of our models. We performed the cross validation across seeds 40 to 49 on all of the datasets. Moreover, we first used all continuous and categorical columns, then we used just continuos columns excluding the longitude and latitude, and lastly we used only the highly correlated columns with the targets.

We then evaluated the best models using metrics like mean squared error. Furthermore, we performed the evaluation similarly to the cross validation to reduce the influence of the data set splits. The evaluation metrics were then selected as the median values from the 10 different evaluation runs.

# Models (slides 7 and 8)
We used 4 different regression models. 

A linear regression model as a baseline to assess performance of the other models. 

Second selected model was a fully connected neural network with 2 to 4 hidden layers with ReLU hidden activation function and a linear output activation function. We used the Adam optimizer for an average number of epochs recorded during the cross validation. Consequently, we used early stopping as the stopping criterion during cross validation.

TODO

TODO

# Prediction Performance, Ensemble (slide 9)
TODO

In the end we also used ensemble modeling to combine the results of all 4 models to create a more precise result. Therefor we took
the mean over all 4 models which results in our final prediction result.
