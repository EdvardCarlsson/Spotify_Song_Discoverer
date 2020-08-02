# Spotify Song Discoverer

## Introduction
During big parts of my days I listen to music through Spotify and I'm always on the search for something new. Spotify do today generate quite a lot of recommended songs, with features like the "Discover weekly" and "Release Radar" playlists but I though that i should try it out myself and see if what whose algorithms work the best. I also saw it as a fun opportunity to improve my python and ML skills as well as learn how to handle APIs while at the same time build something which is accually useful to me. 

### Special Acknowledgment
Thanks to Spotify both for providing a great service and for providing__ an easy-to-use web API with helpful guides.

And to the youtube channel CodingEntrepreneurs for guiden...

https://www.youtube.com/c/CodingEntrepreneurs

## Objective
The goal of this project is to create a classifier which, based on my recent listening history, will recommend me new songs. 

My plan was to collect all the song which I've listen to during the last week. Rank them by most played artist and choose the top 10. Then choose each of these artists top 10 related artists. This gives 100 profiles too choose from. From these i pick their top 10 most popular songs  to my dataset of possible songs. 1000 in total, minus a few dupicates and some which filters away because i've already listned to them.

## Data
### Gathering of the Data
I'm using the last.fm application connected to my spotify account to record my listening history, from their website i collect a csv file of the history. For each recorded song I recive it's title, the artist, the album name and the time it was played. For the week this was built I had 547 recordings, which when accounted for dublicates, was 289 different songs. This is the data I will be using when building the classification model.

The sample of potential songs are called from Spotifys web API using json requests. As earlier mentioned i colleced 1000 songs which, based on this week's sample, filtered down to 816. In future weeks, as the number of songs I've listned to increases, more songs will get removed and the need to increase the base of potential songs might become apparent.

### Audio Features
Spotifys API offers something called Audio Features

## Summary Statistics

lite tabeller om hur mycket av varje det är



## Feature Engineering

genres
top artist

## Classification
### Feature Selection
corr_matrix
PCA
### Very Unbalanced Dataset
smote
### Hyperparameter tuning 
3.5% better prec compated to base model

## Recommendations & Conlcussions

## Futher Possible Extensions

