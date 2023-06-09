# Introduction (slide 1)
Dear audience, welcome to our presentation. My name is X and together with my colleagues Y and Z we will discuss regression. Specifically, we will focus on data set pre-processing and on prediction with different regression models.

# Outline (slide 2)
The outline of the presentation as well as how we have approached the project is the following. First, we have researched information about the Universities data set and performed some analysis. We have followed the analysis by cleaning and pre-processing the data set. Then we have defined a training and evaluation scheme used across different models we have tested. Lastly, we have analyzed the performance of the chosen models and created an ensemble to see if they can improve together.

# Data Set Analysis (slide 3)
TODO

# Cleaning and Pre-processing (slide 5)
Cleaning and pre-processing is the most important part of our project. We have used a 6 stage pipeline for it. 

The pipeline starts with identification of missing values, which are not directly NaN, for example the values 9999 in the founded year column. We have also splitted compound columns like the academic staff column and converted percentage columns to floating point values. We have also included deduplication, removal of duplicit rows that is, at the end of the 1st stage.

The second stage is very simple, the data set is splitted to train and test set 80 to 20. It is important to make the data set split at this point in order to prevent leakage of information from the test set to the train set.

We have followed up the second stage with missing value imputation. We have imputed the now splitted data set using 3 different approaches. First, we have used mean imputation of the continuous columns and mode imputation of the categorical columns. Second, we have just replaced mean with median and third, we have used median, mode, linear regression and K-nearest-neighbors imputation. With all approaches we have made sure to impute the missing values only with statistics obtained from the train set to not leak the test set into the train set as already mentioned.

The fourth stage was normalization. We have normalized all the continuos columns to zero mean and unit variance, because the selected models perform better, when the input values are centered around zero. We haven't normalized the predicted columns, which have chosen to be the under graduate average fees and post graduate average fees. And again, we have made sure to normalize all the data sets just with statistics obtained from the train data sets.

Next, we have preformed one-hot encoding of the categorical features.

And in the last stage we have removed all non-numeric columns like the university name and motto.

# Training and Evaluation (slide 6)
We have developed a training and evaluation schema, which we have then applied to all of our models. We have used 3-fold cross validation, since the data set is very small, to find the best best hyper-parameters of our models. We performed the cross validation with seed 14, which is our group number. Moreover, we have performed the cross validation on 3 subsets of the mean, median and mixed imputed data sets. We first used all continuous and categorical columns, then we have used just continuos columns excluding the longitude and latitude columns, and the last subset contained only highly correlated columns with the predicted columns.

We have then evaluated the models selected based on the cross validation using metrics like mean squared error, mean absolute error, root mean squared error and R2 score. Furthermore, we have performed the evaluation as an average across 5 different seeds, as the data set is very small and results are greatly influenced by the data set split.

# Models (slides 7 and 8)
We have used 4 different regression models. 

A linear regression model as a baseline to assess performance against. 

Second selected model was a fully connected neural network with 2 to 4 hidden layers with ReLU activation function and a linear output activation function. We have used the Adam optimizer for training for an average number of epochs recorded during the cross validation. Consequently, we have used early stopping as the stopping criterion during cross validation.

TODO

TODO

# Prediction Performance, Ensemble (slide 9)
TODO
