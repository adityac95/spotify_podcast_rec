# Spotify podcast recommendation engine
by Aditya Chander, Ritika Khurana, Taylor Mahler and Yuchen Luo

We built a podcast recommendation engine that suggests episodes to a listener based on either a previous episode that they've heard or an episode description that they can input with freeform text entry. This project was built for the Erdos Institute Data Science bootcamp, Fall 2022.
***

In the README, we describe the data gathering process, the preprocessing and cleanup, the architecture of the classifier (along with its flaws), and the web-frontend ([link](http://app.sayantankhan.io/search) to web-frontend).

## Table of contents
1. [Data gathering](#data-gathering)
2. [Preprocessing and cleanup](#preprocessing)
3. [Encoding the transcripts](#encoding)
	1. [TFIDF and MiniLM-L6-v2](#options)
	2. [Evaluation](#evaluation)
4. [Streamlit app](#streamlit)

## Data gathering <a name="data-gathering"></a>

We used a [podcast dataset provided by Spotify](https://podcastsdataset.byspotify.com/).[^1] This dataset contains over 200,000 podcast episodes with associated audio, transcripts and metadata. We worked with a subset of around 40,000 podcasts for this project, using only the transcripts and metadata.

[^1]: The data can be requested at the linked website; we cannot provide the original data on Github. 

## Preprocessing and cleanup <a name="preprocessing"></a>

The transcripts were stored in .json files 

## Encoding the transcripts <a name="encoding"></a>

### TFIDF and MiniLM-L6-v2 <a name="options"></a>

### Evaluation <a name="evaluation"></a>

## Streamlit app <a name="streamlit"></a>

The front-end is built in [Flask](https://flask.palletsprojects.com/en/2.1.x/).
To build and run the web-server, navigate to [web_interface](web_interface), and run the following commands (from the project root directory) to reconstruct the data files and set up and start the server.
```
sh scripts/reconstruct_large_data_files.sh
poetry env use python3.8
poetry install
poetry shell
flask run
```
Click on [this link](http://app.sayantankhan.io/search) to go to the hosted version of the web-interface.
