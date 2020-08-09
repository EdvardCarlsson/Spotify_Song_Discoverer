# Spotify Song Discoverer

This README is still being written, but the code in the other file is working

## Introduction
During big parts of my days, I listen to music through Spotify and I'm always on the search for something new. Spotify does today generate quite a lot of recommended songs, with features like the "Discover Weekly" and "Release Radar" playlists but I thought that I should try it out myself and see if what whose algorithms work the best. I also saw it as a fun opportunity to improve my python and ML skills as well as learn how to handle APIs while at the same time build something which is actually useful to me. 

### Special Acknowledgment
I would like to thanks to Spotify both for having a great service in general and for proving with an easy-to-use web API with nice guides. And also the youtube channel CodingEntrepreneurs for the helpful guide for working with the Spotify API https://www.youtube.com/c/CodingEntrepreneurs.

## Objective
The goal of this project is to create a classifier that, based on my recent listening history of last week, will recommend new songs. 

My plan was to collect all the songs which I've listened to during the last week. Rank them by most played artists and choose the top 10. Then choose each of these artists, the top 10 related artists. This gives 100 profiles to choose from. From there, I pick their top 10 most popular songs to my dataset of possible songs. 1000 in total, minus a few duplicates and some which filter away because I've already listened to them.

## Data
### Gathering of the Data
I'm using the last.fm application connected to my Spotify account to record my listening history, from their website I collect a csv file of the history. For each recorded song I received its title, the artist, the album name and the time it was played. For the week this was built I had 547 recordings, which when accounted for duplicates, were 289 different songs. This is the data I will be using when building the classification model.

The sample of potential songs is called from Spotify's web API using JSON requests. As earlier mentioned I collected 1000 songs which, based on this week's sample, filtered down to 816. In future weeks, as the number of songs I've listened to increases, more songs will get removed and the need to increase the base of potential songs might become apparent.

### Audio Features
Spotify's API offers something called Audio Features

## Summary Statistics




## Feature Engineering

genres
top artist

## Classification
### Feature Selection
corr_matrix
PCA
### Very Unbalanced Dataset
smote
### Random Forest
### K-Nearest Neighbor

## Recommendations & Conlcussions

I've now run my Song Discover over the consecutive weeks. It has become clear that it does perform better when the music I've listened to during the week is more uniform, mostly focused around a few genres. This was the case during the first and third weeks but not the second. When the learning sample was more diverse the classifier gave me a lot more songs as recommendations. Recommendations, which according to me, were considerably less precise to my taste compared to the other weeks. The code and shown results are based on the third week.
During the third week I only got six songs recommended. This can largely be contributed to the fact that my total listening history is getting longer and therefore more songs are getting filtered out as already played. Since I would rather see a larger number of songs getting recommended I'm considering increasing the sample of potential songs, which in the current state is less than 1000.

## Further Possible Extensions
