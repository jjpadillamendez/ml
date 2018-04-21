# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:08:19 2018

@author: Jesus
"""

# Looking at the vocabulary in detail
#feature_names = vect.get_feature_names()
#print("Number of features: {}".format(len(feature_names)))
#print("First 20 features:\n{}".format(feature_names[:20]))
#print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
#print("Every 2000th feature:\n{}".format(feature_names[::2000]))
#
#from sklearn.model_selection import GridSearchCV
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
#grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
#grid.fit(X_train, y_train)
#print("Best cross-validation score: {:.2f}".format(grid.best_score_))
#print("Best parameters: ", grid.best_params_)

#from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
#print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.pipeline import make_pipeline
#pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None),
#LogisticRegression())
#param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
#grid = GridSearchCV(pipe, param_grid, cv=5)
#grid.fit(text_train, y_train)
#print("Best cross-validation score: {:.2f}".format(grid.best_score_))



#vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
## transform the training dataset
#X_train = vectorizer.transform(text_train)
## find maximum value for each of the features over the dataset
#max_value = X_train.max(axis=0).toarray().ravel()
#sorted_by_tfidf = max_value.argsort()
## get feature names
#feature_names = np.array(vectorizer.get_feature_names())
#print("Features with lowest tfidf:\n{}".format(
#feature_names[sorted_by_tfidf[:20]]))
#print("Features with highest tfidf: \n{}".format(
#feature_names[sorted_by_tfidf[-20:]]))


#pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
## running the grid search takes a long time because of the
## relatively large grid and the inclusion of trigrams
#param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
#"tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
#grid = GridSearchCV(pipe, param_grid, cv=5)
#grid.fit(text_train, y_train)
#print("Best cross-validation score: {:.2f}".format(grid.best_score_))
#print("Best parameters:\n{}".format(grid.best_params_))

# extract feature names and coefficients
#vect = grid.best_estimator_.named_steps['tfidfvectorizer']
#feature_names = np.array(vect.get_feature_names())
#coef = grid.best_estimator_.named_steps['logisticregression'].coef_
#mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)


## extract scores from grid_search
#scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
## visualize heat map
#heatmap = mglearn.tools.heatmap(
#scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
#xticklabels=param_grid['logisticregression__C'],
#yticklabels=param_grid['tfidfvectorizer__ngram_range'])
#plt.colorbar(heatmap)

#mglearn.tools.visualize_coefficients(
#grid.best_estimator_.named_steps["logisticregression"].coef_,
#feature_names, n_top_features=40)


#vect = CountVectorizer(max_features=10000, max_df=.15)
#X = vect.fit_transform(text_train)

#from sklearn.decomposition import LatentDirichletAllocation
#lda = LatentDirichletAllocation(n_topics=10, learning_method="batch",
#max_iter=25, random_state=0)
## We build the model and transform the data in one step
## Computing transform takes some time,
## and we can save time by doing both at once
#document_topics = lda.fit_transform(X)



## For each topic (a row in the components_), sort the features (ascending)
## Invert rows with [:, ::-1] to make sorting descending
#sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
## Get the feature names from the vectorizer
#feature_names = np.array(vect.get_feature_names())
#
## Print out the 10 topics:
#mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
#sorting=sorting, topics_per_chunk=5, n_words=10)

#from sklearn.feature_extraction.text import TfidfVectorizer
#vect = TfidfVectorizer(min_df=5, norm=None)
#

# Is the dataset balanced?
#print("Samples per class (training): {}".format(np.bincount(df['project_is_approved'])))