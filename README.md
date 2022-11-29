# Spotify podcast recommendation engine
by Aditya Chander, Ritika Khurana, Taylor Mahler and Yuchen Luo

We built a podcast recommendation engine that suggests episodes to a listener based on either a previous episode that they've heard or an episode description that they can input with freeform text entry. This project was built for the Erdos Institute Data Science bootcamp, Fall 2022.
***

In the README, we describe the data gathering process, the preprocessing and cleanup, the chosen encodings of the pocasts, and the Streamlit app.

## Table of contents
1. [Data gathering](#data-gathering)
2. [Preprocessing and cleanup](#preprocessing)
	1. [Transcripts](#transcripts)
	2. [Podcast categories](#categories)
3. [Encoding the transcripts](#encoding)
	1. [TFIDF and MiniLM-L6-v2](#options)
	2. [Evaluation](#evaluation)
4. [Streamlit app](#streamlit)

## Data gathering <a name="data-gathering"></a>

We used a [podcast dataset provided by Spotify](https://podcastsdataset.byspotify.com/).[^1] This dataset contains over 200,000 podcast episodes in English and Portuguese with associated audio, transcripts and metadata. We worked with a subset of around 40,000 English-language podcasts for this project, using only the transcripts and metadata.

[^1]: The data can be requested at the linked website; we cannot provide the original data on Github. 

## Preprocessing and cleanup <a name="preprocessing"></a>

The majority of transcript cleaning and category extraction took place in [this notebook](https://github.com/adityac95/erdos_spotify_podcast_rec/blob/main/data_inspection_cleaning_CLEAN.ipynb). 

### Transcripts <a name="transcripts"></a>

The provided transcripts were stored in JSON files. The JSON files contained chunks of text that were automatically transcribed from the audio, together with an estimate of the confidence in the transcription, word-level time alignment, and (on occasion) alternative transcriptions. For each podcast episode, we extracted the subsections of the transcript and concatenated them together to generate the full text.

### Podcast categories <a name="categories"></a>

Each show in the podcast dataset came with RSS feeds in XML format. These RSS feeds contained a lot of metadata, including the categories that the podcaster assigned to the show (which were chosen according to a set of categories provided by [Apple](https://podcasts.apple.com/us/genre/podcasts/id26). These categories were extracted using the `BeautifulSoup` library.

The categories themselves contained widely varying numbers of shows and were of differing levels of granularity. However, the iTunes categorisation system is hierarchical (for instance, baseball and basketball podcasts are listed under the "Sports" category). Mapping the granular subcategories from the original data to the parent categories in Apple resulted in a reduction in the number of categories from 117 to 19, a far more tractable number for our purposes. This recategorisation procedure is detailed in [this notebook](https://github.com/adityac95/erdos_spotify_podcast_rec/blob/main/transcript_tagging_embedding_CLEAN.ipynb).

## Encoding the transcripts <a name="encoding"></a>

### TFIDF and MiniLM-L6-v2 <a name="options"></a>

We explored two different encodings for our podcast transcripts: term frequency-inverse document frequency (TFIDF) scores and transcript embeddings from a transformer model (MiniLM-L6-v2).[^2] The TFIDF weights were computed for the top ??? most commonly occurring words in the dataset, excluding stopwords. The 384-dimensional MiniLM-L6-v2 embeddings were computed using the pretrained model provided by the [`sentence_transformers`](https://www.sbert.net/) library.

[^2]: Information about the MiniLM model is available [here](https://arxiv.org/pdf/2002.10957.pdf).

[This notebook](TODO:REPLACE) generates the TFIDF weights, and [this notebook](https://github.com/adityac95/erdos_spotify_podcast_rec/blob/main/transcript_tagging_embedding_CLEAN.ipynb) generates the MiniLM-L6-v2 embeddings.

### Evaluation <a name="evaluation"></a>

We reasoned that a good encoding would on average rate episodes from different categories as *less similar to each other* compared to episodes from within a category. The higher the proportion of category pairs for which this is the case, the better the encoding. For both TFIDF and MiniLM-L6-v2, we performed the following steps:

1. We sampled 50 random podcast episodes from each of the 19 categories.
2. For every *ordered* pair of categories $A$ and $B$, we computed the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between every episode in $A$ with every episode in $B$ (between-category similarities), as well as every episode in $A$ with every other episode in $A$ (within-category similarities).
3. The between- and within-category similarity scores were compared using a one-tailed *t*-test to examine whether the between-category scores were significantly lower than the within-category scores. 

For the MiniLM-L6-v2 embeddings, **88.3%** of ordered category pairs had lower between-category than within-category similarity scores. For the TFIDF weights, only **75.1%** of ordered category pairs met this criterion. Thus, we chose to encode our podcasts using the MiniLM-L6-v2 embeddings for the recommender app.

## Streamlit app <a name="streamlit"></a>

The front-end is built using [Streamlit](https://streamlit.io/). The code for the app is [here](https://github.com/adityac95/erdos_spotify_podcast_rec/blob/main/app.py).

There are two ways the app can be used:
1. A user can specify an episode they've already heard and they will receive recommendations for up to 20 similar podcast episodes from the same category, excluding other episodes from the same show. This is achieved by indexing into pre-generated cosine similarity tables and filtering by the show ID.
2. A user can type in a search query for an episode and they will receive up to 20 recommendations for episodes from all possible categories. This is achieved by generating the embedding for the query, $e_q$, computing the cosine similarity between $e_q$ and all ~40,000 podcasts, and returning the episodes with the highest similarity scores.

A demonstration of the app is available [here](TODO:YOUTUBE_LINK). Currently this app is not available on a web server; we are working to deploy it! 
