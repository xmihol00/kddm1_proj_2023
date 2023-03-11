## IMDB.csv (recommendation)
### David
* It is quite, might be time consuming to **clean** it and **train** ML models.
* We would have to create fictive user with some stats, e.g. age, sex, favorite genres etc., to which , we would recommend. And I don't know how we would rate our model if the recommendation is good/bad, there is no ground truth.
* I personally don't know much about recommendation.

## Undergrad.csv (clustering)
### David
* It is decently large, but lot of missing or wrong values, it would probably need a lot of **cleaning**.
* We would probably have to somehow encode (convert to numbers) the non-numeric colums in order to use them for clustering.
* I don't perticaullary like this project idea, becuase clustering is unsupervised and we can't really tell if it performs well or not.

## Olimpics.csv (classification)
### David
* It is decently large, missing values values in the `Country` but some wrong character, **cleaning** would not be that probably hard.
* No numeric columns apart from the `Year` column, we would have to encode all of them somehow.
* We could either predict the countries, which are most likely to score a medal, given a disciplin, or given a country, which are the most likely disciplins where it will score a medal.
* We would have to create our own ground truths.
* I quite doubt, it we would achieve good results, since there is usually a fair amount of randomness in who wins a medal.

## Universities.csv (regression)
### David
* It is very small, some missing values, but not in important (numeric) columns. Becuase it is small, we can **check by hand**, that every cleaning/imputation algorithm works as intended. 
* Two possible ground truth columns `UG_average_fees_(in_pounds)` and `PG_average_fees_(in_pounds)`, which probably have quite large correlation, we could predict just one of them or both.
* Lot of numeric columns, which can be used for prediction. Colums could be pre-processed and transformed to zero mean and unit variance.
* We could use techniques like cross validation to tackle the small data set. We could mention it in the report and show, that we understand the issue of a small data set.
* We could build a baseline model (Kern likes baseline models), e.g. simples possible linear NN, and then each of us could build his own model using e.g. SVM, decision tree, random forest, NN with hiddent layers (non linear), etc. We could then compare the models in the report and either select the best performing model, or create an ensemble of all 4 of them. The ensamble should in theory perform the best.
* I think this task/data set is the best from the offered and I'm now strongly in favor of this one.
