# Introduction (slide 1)
Dear audience, welcome to our presentation. My name is X and together with my colleagues Y and Z we will discuss regression. Specifically, we will focus on dataset pre-processing and on prediction with different regression models.

# Outline (slide 2)
The outline of the presentation as well as how we have approached the project is the following. First, we have researched information about the Universities dataset and performed some analysis. We have followed the analysis by cleaning and pre-processing the dataset. Then we have defined a training and evaluation scheme used across different models that we have tested. Lastly, we have analyzed the performance of the chosen models and created an ensemble to see if we cn improve the performance with it.

# dataset Analysis (slide 3)
TODO

# Cleaning and Pre-processing (slide 5)
Cleaning and pre-processing is the most important part of our project. We have used a 6 stage pipeline for it. 

The pipeline starts with identification of missing values, which are not directly NaN, for example the values 9999 in the founded year column. We have also splitted compounded columns like the academic staff column and converted percentage columns to floating point values. We have also included deduplication at the end of the 1st stage.

The second stage is very simple, the dataset is splitted to training and test dataset 80 to 20. It is important to make the dataset split at this point in order to prevent leakage of information from the test dataset to the training dataset.

We have followed up the second stage with missing value imputation. We have imputed the now splitted dataset using 3 different approaches. First, we have used mean imputation of the continuous columns and mode imputation of the categorical columns. Second, we have just replaced mean with median and third, we have used median, mode, linear regression and K-nearest-neighbors imputation on different columns. With all approaches we have made sure to impute the missing values only with statistics obtained from the training dataset to not leak the test dataset into the training dataset as already mentioned.

The fourth stage was normalization. We have normalized all the continuos columns to zero mean and unit variance, because the selected models perform better, when the input values are centered around zero. We haven't normalized the predicted columns, which we have chosen to be the **under graduate average fees** and **post graduate average fees**. And again, we have made sure to normalize all the datasets just with statistics obtained from the training dataset.

Next, we have preformed one-hot encoding of the categorical features.

And in the last stage we have removed all non-numeric columns like the university name and motto.

# Training and Evaluation (slide 6)
We have developed a training and evaluation schema, which we have then applied to all of our models. To find the best hyper-parameters  we have used 3-fold cross validation, since the dataset is very small. We performed the cross validation with seed 14, which is our group number. Moreover, we have performed the cross validation on 3 subsets of the mean, median and mixed imputed datasets. We first used all continuous and categorical columns, then we have used just continuos columns excluding the longitude and latitude columns, and the last subset contained only highly correlated columns with the predicted columns.

We have then evaluated the models selected based on the cross validation using metrics like mean squared error, mean absolute error, root mean squared error and R2 score. Furthermore, we have performed the evaluation as an average across 5 different seeds, as the dataset is very small and the results are greatly influenced by the dataset split.

# Models (slides 7 and 8)
We have used 4 different regression models. 

A linear regression model as a baseline to assess performance of the other models. 

Second selected model was a fully connected neural network with 2 to 4 hidden layers with ReLU hidden activation function and a linear output activation function. We have used the Adam optimizer for an average number of epochs recorded during the cross validation. Consequently, we have used early stopping as the stopping criterion during cross validation.

TODO

TODO

# Prediction Performance, Ensemble (slide 9)
TODO

In the end we also used ensemble modeling to combine the results of all 4 models to create a more precise result. Therefor we took
the mean over all 4 models which results in our final prediction result.
