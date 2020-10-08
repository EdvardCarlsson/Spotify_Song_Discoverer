#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:33:39 2020

@author: edvardcarlsson
"""

from SpotifyAPI import SpotifyAPI 
import pandas as pd
import numpy as np


client_id = "ead07f131586436e88d6ed3f0a07d034"
client_secret = "6cc0d2298bde4885a5400b734fd0d663"
spotify = SpotifyAPI(client_id, client_secret)




def related_artists_10(related_artists):
    related = []
    for k in range(10):    
        artist = {"Namn" : related_artists["artists"][k]["name"],
              "Genre": related_artists["artists"][k]["genres"],
              "Popularity":related_artists["artists"][k]["popularity"],
              "Id":related_artists["artists"][k]["id"],
              "Top_Songs": top_tracks(related_artists["artists"][k]["id"])
            }
        related.append(artist)
    return related 

# tar fram en artists top tracks och dess features
def top_tracks(_id):
    rek1 = spotify.get_artist_top_tracks(_id)
    dvs = pd.DataFrame(columns={"Name","Popularity","Id",
                                "acousticness", "danceability","duration_ms", "energy",
                                "instrumentalness", "key", "liveness", "loudness", "mode", 
                                "speechiness","tempo","time_signature","valence"
                                }, index = range(len(rek1["tracks"])))
    for k in range(len(rek1["tracks"])):
        dvs["Name"][k] = rek1["tracks"][k]["name"]
        dvs["Popularity"][k] = rek1["tracks"][k]["popularity"]
        dvs["Id"][k] = rek1["tracks"][k]["id"]
        audio_features = spotify.get_audio_features(dvs["Id"][k])
        dvs["acousticness"][k] = audio_features["acousticness"]
        dvs["danceability"][k] = audio_features["danceability"]
        dvs["duration_ms"][k] = audio_features["duration_ms"]
        dvs["energy"][k] = audio_features["energy"]
        dvs["instrumentalness"][k] = audio_features["instrumentalness"]
        dvs["key"][k] = audio_features["key"]
        dvs["liveness"][k] = audio_features["liveness"]
        dvs["loudness"][k] = audio_features["loudness"]
        dvs["mode"][k] = audio_features["mode"]
        dvs["speechiness"][k] = audio_features["speechiness"]
        dvs["tempo"][k] = audio_features["tempo"]
        dvs["time_signature"][k] = audio_features["time_signature"]
        dvs["valence"][k] = audio_features["valence"]
        columns_order = ["Name","Id","Popularity",
                "acousticness", "danceability","duration_ms", "energy",
                "instrumentalness", "key", "liveness", "loudness", "mode", 
                "speechiness","tempo","time_signature","valence"]
        dvs=dvs.reindex(columns=columns_order)

    return dvs


använda_id = pd.DataFrame(columns={"Artist","id"},index = range(3))
columns_order =["Artist","id"]
använda_id=använda_id.reindex(columns=columns_order)




def artist_id(song_name,artist_name):
    _id = spotify.search({"track": song_name,"artist": artist_name},
                           search_type ="track")["tracks"]["items"][0]["artists"][0]["id"]
    artist = [artist_name]
    artist.append(_id)
    
    return artist
    


def data_from_spotify(mest_lyssnade):  
    sample_list = []

    for k in range(len(mest_lyssnade)):
        start_artist = []
        related_artists = spotify.get_related_artists(mest_lyssnade.iloc[k,2])
        start_artist = related_artists_10(related_artists)
        sample_list.append(start_artist)


    sample = pd.DataFrame(columns=sample_list[0][0]["Top_Songs"].columns, index = range(1000))
    sample["Artist"] = np.nan
    sample["Genre"] = np.nan
    sample["Artist_Popularity"] = np.nan
   
    n = 0
    m = 0
    for i in range(len(sample_list)):
        n = 100 * i
        for k in range(10):
            m = 10 * k
            for j in range(10):
                if j < len(sample_list[i][k]["Top_Songs"]):
                    sample.iloc[n+m+j] = sample_list[i][k]["Top_Songs"].iloc[j]
                    sample["Artist"][n+m+j] = sample_list[i][k]["Namn"]
                    sample["Genre"][n+m+j] = sample_list[i][k]["Genre"]
                    sample["Artist_Popularity"][n+m+j] = sample_list[i][k]["Popularity"]
        m = 0 
    
    sample = sample.dropna(subset=["Id"])
    sample = sample.reset_index(drop=True)


    return sample





