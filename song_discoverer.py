#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:28:21 2020

@author: edvardcarlsson
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")



client_id = "spotify_client_id"
client_secret = "spotify_client_secret"
spotify = SpotifyAPI(client_id, client_secret)


# 1: Song Samples

# Import of my entire listening history, since this is the third week I try the Song Discoverer and I only want to base the recomendation on last weeks listens it is important that I seperate them from older listens.
listened = pd.read_csv("listened_week_32.csv")

# The first row of my data gets assigned as columns, this fixes it
listened.loc[-1] = [listened.columns[0], listened.columns[1], listened.columns[2], listened.columns[3]]  # adding a row
listened.index = listened.index + 1  # shifting index
listened.sort_index(inplace=True) 
listened.columns = ["Artist", "Album", "Song","Date"]
listened = listened.reset_index(drop=True)
print("Number of songs from all time listening history: ", len(listened))


# Since I can see the recording timestamp of my plays I can locate with songs are from this week, I'm making a new dataframe of them.
nbr_of_songs_this_week = 833
listened_last_week = listened[:nbr_of_songs_this_week]

# I'm also saving this data to a new csv file for possible future extensions of the project
listened_last_week.to_csv(r'week_31.csv')

listened_last_week.head()




# As my algorithm is based on my top 10 most streamed artist from the last week I create a new dataframe with them
top_artists = pd.DataFrame(listened_last_week.groupby(["Artist"]).count())
top_artists["Artist"] = top_artists.index
top_artists = top_artists.sort_values(by="Date", ascending=False)
top_artists = top_artists.iloc[0:10,2:]
top_artists.columns = ["Plays", "Artist"]
top_artists = top_artists.reset_index(drop=True)

top_artists["id"] = np.nan
for k in range(len(top_artists)):
    top_artists["id"][k] = spotify.search({"artist": top_artists.iloc[k,1]}, search_type ="artist")["artists"]["items"][0]["id"]
top_artists




# Gathering my sample of potential songs
potential_songs = data_from_spotify(top_artists)  
potential_songs.head()




# Gather the song features of "listened_last_week"
listened_features = pd.DataFrame(index = listened_last_week.index, columns=potential_songs.columns)
listened_features["Name"] = listened_last_week["Song"]

for k in range(len(listened_last_week)):
    
    if len(spotify.search({"track": listened_last_week.iloc[k,2],"artist": listened_last_week.iloc[k,0],
                "album":listened_last_week.iloc[k,1]}, 
            search_type ="track")["tracks"]["items"]) == 0: # If the song is not found
        continue
    else:
        track = spotify.search({"track": listened_last_week.iloc[k,2],"artist": listened_last_week.iloc[k,0],
                "album":listened_last_week.iloc[k,1]}, 
                search_type ="track")["tracks"]["items"][0]
    
        listened_features["Id"][k] = track["id"]
        listened_features["Popularity"][k] = track["popularity"]
        
        audio_features = spotify.get_audio_features(track["id"])
        listened_features["acousticness"][k] = audio_features["acousticness"]
        listened_features["danceability"][k] = audio_features["danceability"]
        listened_features["duration_ms"][k] = audio_features["duration_ms"]
        listened_features["energy"][k] = audio_features["energy"]
        listened_features["instrumentalness"][k] = audio_features["instrumentalness"]
        listened_features["key"][k] = audio_features["key"]
        listened_features["liveness"][k] = audio_features["liveness"]
        listened_features["loudness"][k] = audio_features["loudness"]
        listened_features["mode"][k] = audio_features["mode"]
        listened_features["speechiness"][k] = audio_features["speechiness"]
        listened_features["tempo"][k] = audio_features["tempo"]
        listened_features["time_signature"][k] = audio_features["time_signature"]
        listened_features["valence"][k] = audio_features["valence"]
        
        listened_features["Artist"][k] = listened_last_week.iloc[k,0]
        artist_id_ = track["artists"][0]["id"]
        artist_features = spotify.get_artist(artist_id_)
        listened_features["Genre"][k]  = artist_features["genres"]
        listened_features["Artist_Popularity"][k]  = artist_features["popularity"]

# Some songs are not found through my search. I´m dropping these, it's unfortunate but since I still have a quite large sample it will do
listened_features = listened_features.dropna(subset=["Id"])
listened_features = listened_features.reset_index(drop=True)


# Save into this df so we only need to find features of last weeks songs
# we only need id
listened_all_time = pd.read_csv("listened_all_time.csv")
listened_all_time.append(listened_features["Id"])
listened_all_time.to_csv(r'listened_all_time.csv')


listened_features.head()


# Adding number of plays and creating a new dataframe with my 10 most played songs
listened_features["Plays"] = listened_features["Name"].apply(lambda x: (listened_features["Name"] == x).sum())
listened_features.drop_duplicates(subset ="Name",keep = "first", inplace = True) 

most_listened_songs = listened_features.sort_values(by="Plays", ascending=False).iloc[0:10]
most_listened_songs = most_listened_songs.reset_index(drop=True)   
for k in range(len(most_listened_songs)):
    print(most_listened_songs["Name"][k],"| Plays:", most_listened_songs["Plays"][k])



# It shouldn't be possible for the program to recommend song which I've already heard, i.e. appers on "listened". Also duplicates are filtered away
potential_songs = potential_songs[~potential_songs["Id"].isin(listened_all_time["Id"])]
potential_songs.drop_duplicates(subset ="Id",keep = "first", inplace = True) 
potential_songs = potential_songs.reset_index(drop=True)   
print("Number of potential songs: ", len(potential_songs))


# 2 Summary Statistics

# Feature Statistics
fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["Popularity"], color="g", ax=ax[0], kde=False).set_title("Popularity (possible songs)")
sns.distplot(listened_features["Popularity"], color="y", ax=ax[1], kde=False).set_title("Popularity (listened songs)")
sns.distplot(most_listened_songs["Popularity"], color="r", ax=ax[2], kde=False).set_title("Popularity (10 most listened songs)")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["acousticness"], color="g", ax=ax[0], kde=False).set_title("Acousticness")
sns.distplot(listened_features["acousticness"], color="y", ax=ax[1], kde=False).set_title("Acousticness")
sns.distplot(most_listened_songs["acousticness"], color="r", ax=ax[2], kde=False).set_title("Acousticness")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["danceability"], color="g", ax=ax[0], kde=False).set_title("Danceability")
sns.distplot(listened_features["danceability"], color="y", ax=ax[1], kde=False).set_title("Danceability")
sns.distplot(most_listened_songs["danceability"], color="r", ax=ax[2], kde=False).set_title("Danceability")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["duration_ms"], color="g", ax=ax[0], kde=False).set_title("Duration (ms)")
sns.distplot(listened_features["duration_ms"], color="y", ax=ax[1], kde=False).set_title("Duration (ms)")
sns.distplot(most_listened_songs["duration_ms"], color="r", ax=ax[2], kde=False).set_title("Duration (ms)")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["energy"], color="g", ax=ax[0], kde=False).set_title("Energy")
sns.distplot(listened_features["energy"], color="y", ax=ax[1], kde=False).set_title("Energy")
sns.distplot(most_listened_songs["energy"], color="r", ax=ax[2], kde=False).set_title("Energy")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["instrumentalness"], color="g", ax=ax[0], kde=False).set_title("Instrumentalness")
sns.distplot(listened_features["instrumentalness"], color="y", ax=ax[1], kde=False).set_title("Instrumentalness")
sns.distplot(most_listened_songs["instrumentalness"], color="r", ax=ax[2], kde=False).set_title("Instrumentalness")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["key"], color="g", ax=ax[0], kde=False).set_title("Key")
sns.distplot(listened_features["key"], color="y", ax=ax[1], kde=False).set_title("Key")
sns.distplot(most_listened_songs["key"], color="r", ax=ax[2], kde=False).set_title("Key")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["liveness"], color="g", ax=ax[0], kde=False).set_title("Liveness")
sns.distplot(listened_features["liveness"], color="y", ax=ax[1], kde=False).set_title("Liveness")
sns.distplot(most_listened_songs["liveness"], color="r", ax=ax[2], kde=False).set_title("Liveness")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["loudness"], color="g", ax=ax[0], kde=False).set_title("Loudness")
sns.distplot(listened_features["loudness"], color="y", ax=ax[1], kde=False).set_title("Loudness")
sns.distplot(most_listened_songs["loudness"], color="r", ax=ax[2], kde=False).set_title("Loudness")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["mode"], color="g", ax=ax[0], kde=False).set_title("Mode")
sns.distplot(listened_features["mode"], color="y", ax=ax[1], kde=False).set_title("Mode")
sns.distplot(most_listened_songs["mode"], color="r", ax=ax[2], kde=False).set_title("Mode")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["speechiness"], color="g", ax=ax[0], kde=False).set_title("Speechiness")
sns.distplot(listened_features["speechiness"], color="y", ax=ax[1], kde=False).set_title("Speechiness")
sns.distplot(most_listened_songs["speechiness"], color="r", ax=ax[2], kde=False).set_title("Speechiness")

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["tempo"], color="g", ax=ax[0], kde=False).set_title("Tempo")
sns.distplot(listened_features["tempo"], color="y", ax=ax[1], kde=False).set_title("Tempo")
sns.distplot(most_listened_songs["tempo"], color="r", ax=ax[2], kde=False).set_title("Tempo")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["time_signature"], color="g", ax=ax[0], kde=False).set_title("Time Signature")
sns.distplot(listened_features["time_signature"], color="y", ax=ax[1], kde=False).set_title("Time Signature")
sns.distplot(most_listened_songs["time_signature"], color="r", ax=ax[2], kde=False).set_title("Time Signature")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["valence"], color="g", ax=ax[0], kde=False).set_title("Valence")
sns.distplot(listened_features["valence"], color="y", ax=ax[1], kde=False).set_title("Valence")
sns.distplot(most_listened_songs["valence"], color="r", ax=ax[2], kde=False).set_title("Valence")
plt.show()

fig, ax = plt.subplots(1,3,figsize=(12,3))
sns.distplot(potential_songs["Artist_Popularity"], color="g", ax=ax[0], kde=False).set_title("Artist_Popularity")
sns.distplot(listened_features["Artist_Popularity"], color="y", ax=ax[1], kde=False).set_title("Artist_Popularity")
sns.distplot(most_listened_songs["Artist_Popularity"], color="r", ax=ax[2], kde=False).set_title("Artist_Popularity")
plt.show()


# Genres
genre_sum = genres(potential_songs, listened_features, most_listened_songs)

top_genres = genre_sum.sort_values(by="Top", ascending=False).loc[genre_sum["Top"] != 0]
genre_match_list = top_genres["Genre"][0:7].values.tolist() 


print("My most frequent listened genres, ordered by occurrence in most_listened_songs")
top_genres[0:7]


# 3. Feature Engineerning

listened_features = listened_features.reset_index(drop=True)

# top genre, my top 7 genres this week
listened_features["genre_match"] = 0
for k in range(len(listened_features)):   
    lista = listened_features["Genre"][k]
    if any(genre in lista for genre in genre_match_list):
        listened_features["genre_match"][k] = 1
        
potential_songs["genre_match"] = 0
for k in range(len(potential_songs)):
    if len(potential_songs["Genre"][k]) != 0:
        lista =  potential_songs["Genre"][k]
        if any(genre in lista for genre in genre_match_list):
            potential_songs["genre_match"][k] = 1


# top artist as a feature my top 10 most heavily listen artiist
listened_features["Top_artist"] = 0                    
top_artist_list = top_artists["Artist"].values.tolist()
for k in range(len(listened_features)):
    if any(artist == listened_features["Artist"][k] for artist in top_artist_list):
        listened_features["Top_artist"][k] = 1

potential_songs["Top_artist"] = 0                    
top_artist_list = top_artists["Artist"].values.tolist()
for k in range(len(potential_songs)):
    if any(artist == potential_songs["Artist"][k] for artist in top_artist_list):
        potential_songs["Top_artist"][k] = 1


listened_features = listened_features.sort_values(by="Plays", ascending=False)
listened_features["Top_Song"] = 0
listened_features["Top_Song"][0:10] = 1        
        
# save these for possible extension work
listened_features.to_csv(r"songs_features_week_32.csv")



# 4 Classification

from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score ,roc_auc_score, f1_score, confusion_matrix


data = pd.read_csv("songs_features_week_32.csv")
data = data.drop(columns="Unnamed: 0")

# Dropping the least interesting columns and the ones that are completely arbitrary columnerna. Also dropping the top_artist column since it will have a too strong influence on the classification and lead to that only these songs get recommended
data = data.drop(columns=["Name","Id","key","mode","time_signature","Artist","Genre","Plays","Top_artist"], axis = 1)
data.shape


# Correlation matrix for feature selection
plt.figure(figsize=(16,16))
corr = sns.heatmap(np.round(data.corr(),2),annot=True, center=0,linewidths=.5)

# There are strange cut offs for the highest/lowest rows, fixed with the following 
bottom, top = corr.get_ylim()
corr.set_ylim(bottom + 0.5, top - 0.5)


# In preperation for the models I scale all variables
scaler = MinMaxScaler()

data_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

data_scaled_x = data_scaled.drop(columns=["Top_Song"])


# In trying to reduce complexity of the models I want to use fewer variables. By constructing PCAs I want to see how much of the varience is explained by variables used.  
pca=PCA()

pca.fit(data_scaled_x)
data_pca=pca.transform(data_scaled_x)
el=np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=el,color=sns.color_palette("Set2")[0])
ax.set(xlabel='Dimensions', ylabel='Explained Variance Ratio')
plt.title("PCA-transformed cumulated variance explained")
plt.show()

# It turns out we want to use as many columns as possible. My initial idea was to choose the five or so variables with the highest correlation with "Top_Song" but that would give a less clear picture. Also all varibales have quite similar magnitude of correlation so the decion of which to choose could easily become misleading.

# Unbalanced dataset

data_y = data_scaled["Top_Song"]
data_y.value_counts()


# This would create problems for the fitting of the prediction model. Becouse of this I'm oversampling the Top Songs using SMOTE 
oversample = SMOTE()
smote_x, smote_y = oversample.fit_resample(data_scaled_x, data_y)


# Splitting the dataset into 70/30 train/validation set
x_train_70, x_test_30, y_train_70, y_test_30 = train_test_split(smote_x, smote_y, test_size = 0.30, random_state = 42)



#Random Forrest

# Hyperparameter tuning for RandomForest
# Create a parameter grid to sample from during fitting

# Parameters:

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to narrow down the possible best hyperparameters
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(x_train_70, y_train_70)


# Found best parameters
rf_random.best_params_



# Find the specific best parameters by searching every combination close to our earlier findings

# Create a new parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10,20,30],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [1, 2, 3],
    'n_estimators': [800, 1000, 1200]
}

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(x_train_70, y_train_70)

grid_search.best_params_

# Validation of RF 
# Using F1, accuracy, and AUC

rf_best = RandomForestClassifier(n_estimators = 800, min_samples_split = 2,
                                 min_samples_leaf = 1, max_features = 2,
                                 max_depth = 10, bootstrap = False)

rf_best.fit(x_train_70, y_train_70)
rf_pred_30_best = rf_best.predict(x_test_30)

print('Random Forest Model Validation result:\n')
print_confusion_matrix(confusion_matrix(y_test_30,rf_pred_30_best,labels=[1,0]),["Top Song: Yes","Top Song: No"])
print('Accuracy Score: {}\n\n F1 Score: {}\n \n AUC Score: {}'
      .format(accuracy_score(y_test_30,rf_pred_30_best), f1_score(y_test_30,rf_pred_30_best), roc_auc_score(y_test_30,rf_pred_30_best)))


# Comparison with the base classifier
rf_base = RandomForestClassifier()


rf_base.fit(x_train_70, y_train_70)
rf_pred_30_base = rf_base.predict(x_test_30)

print('Base Random Forest Model Validation result:\n')
print_confusion_matrix(confusion_matrix(y_test_30,rf_pred_30_base,labels=[1,0]),["Top Song: Yes","Top Song: No"])
print('Accuracy Score: {}\n\n F1 Score: {}\n \n AUC Score: {}'
      .format(accuracy_score(y_test_30,rf_pred_30_base), f1_score(y_test_30,rf_pred_30_base), roc_auc_score(y_test_30,rf_pred_30_base)))


# The base model turns out to be as good as the optimized model. Somewhat unfortunate the optimization didn't have an impact but at the same time they recive very good scores. In prior weeks I have observed similarly very good scores of the base model but always a small improvement in the optimized.


# K-nearest Neighbor
# Hyperparameter tuning

# Number of neighbors k
n_neighbors = list(range(1,30))

# Leaf size
leaf_size = list(range(1,50))

# Power of for calculating the Minkowski distance
p = list(range(1,4))

random_grid = {
    "n_neighbors": n_neighbors,
    "leaf_size": leaf_size,
    "p": p
}


knn = KNeighborsClassifier()

knn_random = RandomizedSearchCV(estimator = knn, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

knn_random.fit(x_train_70, y_train_70)

knn_random.best_params_


# Create a new parameter grid based on the results of random search 

param_grid = {
    "n_neighbors": [1,2,3,4,5],
    "leaf_size": [27,28,29,30,31],
    "p": [1,2]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(x_train_70, y_train_70)

grid_search.best_params_

# Validation of KNN 

knn_best = KNeighborsClassifier(leaf_size = 27, n_neighbors = 2, p = 1)

knn_best.fit(x_train_70, y_train_70)
knn_pred_30_best = knn_best.predict(x_test_30)

print('Random Forest Model Validation result:\n')
print_confusion_matrix(confusion_matrix(y_test_30,knn_pred_30_best,labels=[1,0]),["Top Song: Yes","Top Song: No"])
print('Accuracy Score: {}\n\n F1 Score: {}\n \n AUC Score: {}'
      .format(accuracy_score(y_test_30,knn_pred_30_best), f1_score(y_test_30,knn_pred_30_best), roc_auc_score(y_test_30,knn_pred_30_best)))


# Comparison with the base classifier
knn_base = KNeighborsClassifier()

knn_base.fit(x_train_70, y_train_70)
knn_pred_30_base = knn_base.predict(x_test_30)

print('Base Random Forest Model Validation result:\n')
print_confusion_matrix(confusion_matrix(y_test_30,knn_pred_30_base,labels=[1,0]),["Top Song: Yes","Top Song: No"])
print('Accuracy Score: {}\n\n F1 Score: {}\n \n AUC Score: {}'
      .format(accuracy_score(y_test_30,knn_pred_30_base), f1_score(y_test_30,knn_pred_30_base), roc_auc_score(y_test_30,knn_pred_30_base)))



X_test = potential_songs.drop(columns=['Name', 'Id',
       'key','mode', 'time_signature',"Genre","Artist","Top_artist"]).astype(float)
    
X_test = pd.DataFrame(scaler.fit_transform(X_test),columns = X_test.columns)

Y_pred_RF = rf_best.predict(X_test)


recommended_songs = pd.DataFrame(index=potential_songs.index)
recommended_songs["Name"] = potential_songs["Name"]
recommended_songs["id"] = potential_songs["Id"]
recommended_songs["Artist"] = potential_songs["Artist"]
recommended_songs["RF_rec"] = Y_pred_RF
recommended_songs = recommended_songs[recommended_songs.RF_rec != 0]
recommended_songs = recommended_songs.reset_index(drop=True)
recommended_songs = recommended_songs.drop(columns=["RF_rec"])
recommended_songs


# To CSV for possible future work, overall I'm quite happy with these recomendations. But maybe a little few, increase potential sample for future weeks
recommended_songs.to_csv(r'reco_week32.csv')




