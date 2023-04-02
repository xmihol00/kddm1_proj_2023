
## IMDB.csv (recommendation)
***
### David
* It is quite, might be time consuming to **clean** it and **train** ML models.
* We would have to create fictive user with some stats, e.g. age, sex, favourite genres etc., to which , we would recommend. And I don't know how we would rate our model if the recommendation is good/bad, there is no ground truth.
* I personally don't know much about recommendation.
### Thomas
* Huge Dataset -> preprocessing (clean, fill gabs with reasonable context etc.) will take some time. 
* I think there is some important information missing for a good recommendation system. E.g. we would like to know something about the persons who voted.
* Also some kind of a summary for the movies would have helped to make at least context based recommendation. But without that I think it is quite boring.
* First idea is to create a user with some history of votes. Based on this votes we could make some recommendations at least for favourite genres, year range, 
movie type etc. So we could restrict the space of recommendations.

### Alexander
* I personally would be interested, but I have no experience with it (hence the interest).
* by using the imdb id you could also scrape erroneous/missing data ( e.g. https://www.imdb.com/title/tt0000002/ ) and user may can be created by userlists (e.g. https://www.imdb.com/list/ls564877733/ not shure if age, sex is very important for this (?,idk) but its may be only theoretical)
</br>

## Undergrad.csv (clustering)
***
### David
* It is decently large, but lot of missing or wrong values, it would probably need a lot of **cleaning**.
* We would probably have to somehow encode (convert to numbers) the non-numeric columns in order to use them for clustering.
* I don't particularly like this project idea, because clustering is unsupervised and we can't really tell if it performs well or not.
### Thomas
* Dataset is in my opinion not that bad, just some cleaning for unreasonable high values would be hard. The values of the other columns should be obvious to complete / change.
* I am also not a friend of clustering, because this would just lead to a biased representation in a way that we could get nice clustered plots to support some argumentations for the dataset.
### Alexander 
* "bigger" dataset (~3900 samples, 392 "useless")
* some high values look like dummy values ( multiple times 9999999 in values and 9999 in year -> maybe easy cleaning) (whereas Arizona == AR?, UT,VT,SC -> can be found in the internet for shure)
* clustering in general maybe sounds interesting what can be found out
* type_{1,2} & Expanse_{1,2} only 4 distinct values (Public,Private / {In,Out}-State)
</br>

## Olympics.csv (classification)
***
### David
* It is decently large, missing values values in the `Country` but some wrong character, **cleaning** would not be that probably hard.
* No numeric columns apart from the `Year` column, we would have to encode all of them somehow.
* We could either predict the countries, which are most likely to score a medal, given a discipline, or given a country, which are the most likely disciplines where it will score a medal.
* We would have to create our own ground truths.
* I quite doubt, it we would achieve good results, since there is usually a fair amount of randomness in who wins a medal.

### Thomas
* The dataset needs some cleaning but the information we need is always in the same row or at least we could predict the right values based on surrounding entries.
* On the first look I could not find a row we could not correct.
* With this dataset we could at least easy check our performance of predictions. (Train/Test split)
* Maybe different training approaches for different predictions. I think we could show/argue how it would impact our results when we just train our model e.g. gender specific or mixed.
* As David already mentioned, there is for sure some kind of randomness, also the performance of some nations have changed over the years. But therefore we could also show this in our summary and make some kind of performance rankings for disciplines e.g. to show the dominance of nations over decades in some disciplines.
* I think there is a huge potential to analyze the dataset and create different approaches, as mentioned in the project remark the performance is not that important but to understand the dataset.

### Alexander
- depends on the task, but it does not look easy to process the data/get usefull information (at a frist glance) . (no real numeric values, many classes (diciplines/sports))
- At first sight, the data seems to be more or less complete.

</br>

## Universities.csv (regression)
***
### David
* It is very small, some missing values, but not in important (numeric) columns. Because it is small, we can **check by hand**, that every cleaning/imputation algorithm works as intended. 
* Two possible ground truth columns `UG_average_fees_(in_pounds)` and `PG_average_fees_(in_pounds)`, which probably have quite large correlation, we could predict just one of them or both.
* Lot of numeric columns, which can be used for prediction. Columns could be pre-processed and transformed to zero mean and unit variance.
* We could use techniques like cross validation to tackle the small data set. We could mention it in the report and show, that we understand the issue of a small data set.
* We could build a baseline model (Kern likes baseline models), e.g. simple possible linear NN, and then each of us could build his own model using e.g. SVM, decision tree, random forest, NN with hidden layers (non linear), etc. We could then compare the models in the report and either select the best performing model, or create an ensemble of all 4 of them. The ensemble should in theory perform the best.
* I think this task/data set is the best from the offered and I'm now strongly in favour of this one.

### Thomas
* In this dataset we could not correct the missing values without using external material (university name, founded year, CWUR_score, Student_satisfaction etc.). Because these are just unknown for us and we cannot reconstruct them out of other data from this dataset. But most likely we will not need them for most of the regression tasks we will just not consider this rows with missing information.
* This dataset offers a huge amount of possibilities for different regression approaches. I think therefore we would not run out of tasks for everyone to contribute within this project. As David already mentioned just alone the amount of possible models and techniques we could use here is for sure a benefit for our final report.


### Alexander
- Many fields can be filled in manually (e.g. University_name, Founded_year), but not shure if CWUR_score is easy to find (~52 entries missing?) 
- dataset has not many samples(~145 entries), could cause problems/be a challenge
- many columns, some not needed (motto?), therefore quite a few regression possibilities
- like in the description mentioned, many possibilities (-> comparisons of the methods/models could be well split up between us, recomender imdb would be harder to split the work) 
</br>

## Ranking
***
| Project                                | Dataset          | Alexander  (*)| David         | Ronald        | Thomas       |
| -------------                          | -------------    | ------------- | ------------- | ------------- |------------- |
| Interactive system/recommender system  | IMDB.csv         | 3/2           | 4             | X             | 4            |
| Clustering                             | Undergrad.csv    | 2/3           | 3             | X             | 3            |  
| Prediction/classification              | Olympics.csv     | 2/3           | 2             | X             | 1            |  
| Prediction/regression                  | Universities.csv | 1             | 1             | X             | 2            | 

(*) most of the tasks have intresting parts, therefore i don
