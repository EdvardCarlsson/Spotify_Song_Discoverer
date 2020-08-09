# Spotify Song Discoverer

## Introduction
During big parts of my days I listen to music through Spotify and I'm always on the search for something new. Spotify do today generate quite a lot of recommended songs, with features like the "Discover weekly" and "Release Radar" playlists but I though that i should try it out myself and see if what whose algorithms work the best. I also saw it as a fun opportunity to improve my python and ML skills as well as learn how to handle APIs while at the same time build something which is accually useful to me. 

### Special Acknowledgment
I would like to thanks to Spotify both for having a great service in general and for proving with an easy-to-use web API with nice guides. And also the the youtube channel CodingEntrepreneurs for the helpful guide for working with the Spotify API https://www.youtube.com/c/CodingEntrepreneurs.

## Objective
The goal of this project is to create a classifier which, based on my recent listening history of last week, will recommend me new songs. 

My plan was to collect all the song which I've listen to during the last week. Rank them by most played artist and choose the top 10. Then choose each of these artists top 10 related artists. This gives 100 profiles too choose from. From these i pick their top 10 most popular songs  to my dataset of possible songs. 1000 in total, minus a few dupicates and some which filters away because i've already listned to them.

## Data
### Gathering of the Data
I'm using the last.fm application connected to my spotify account to record my listening history, from their website i collect a csv file of the history. For each recorded song I recive it's title, the artist, the album name and the time it was played. For the week this was built I had 547 recordings, which when accounted for dublicates, was 289 different songs. This is the data I will be using when building the classification model.

The sample of potential songs are called from Spotifys web API using json requests. As earlier mentioned i colleced 1000 songs which, based on this week's sample, filtered down to 816. In future weeks, as the number of songs I've listned to increases, more songs will get removed and the need to increase the base of potential songs might become apparent.

### Audio Features
Spotifys API offers something called Audio Features

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

I've now run my Song Discover over the consecutive weeks. It has become clear that it does perform better when the music I've listen to during the week is more uniform, mostly focused around a few genres. This was the case during the first and third week but not the second. When the learning sample was more diverse the classifier gave me a lot more songs as recommendations. Recommendations, which according to me, was considerable less precise to my taste compared to the other weeks. The code and shown result are based on the third week.

## Futher Possible Extensions

