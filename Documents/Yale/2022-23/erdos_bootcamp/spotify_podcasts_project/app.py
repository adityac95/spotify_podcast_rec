import streamlit as st
import json
import pandas as pd
import numpy as np
import os
from stqdm import stqdm
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers

@st.cache(suppress_st_warning=True, hash_funcs={dict: lambda _: None, pd.DataFrame: lambda _: None}, allow_output_mutation=True)
def data_loading():
    '''
    Returns a tuple containing:
    * The metadata, `metadata_df`
    * A dictionary of cosine similarity matrices for every category, cosine_sim_matrices
    * A dictionary of matrices indicating whether episodes i and j are from the same show, filter_matrices
    * A dictionary of matrices with the order of shows and episodes in the cosine sim dataframe, row_nums
    * A dictionary of show ids and which category of podcast they are, show_id_to_category
    * A dictionary of show ids and their name, show_id_to_name
    * A dictionary of episode ids and their name, episode_id_to_name
    * The sentence transformer model
    * All of the transcript embeddings
    
    This relies on a particular file structure which can be inferred from the function; however everything can be changed.
    '''
    test = ''
    metadata_df = pd.read_csv('subtranche_metadata.tsv', sep='\t', index_col=0)

    cosine_sim_matrices = {}
    filter_matrices = {}
    row_nums = {}
    embeddings = {}
    
    cosine_sim_dir = 'cosine_sims/'
    filter_table_dir = 'is_same_show/'
    row_nums_dir = 'row_nums/'
    embeddings_dir = 'category_embedding_matrices/'
    
    for cosine_sim_matrix in stqdm(os.listdir(cosine_sim_dir), desc='loading podcast data...hold tight!'):
        if cosine_sim_matrix[-4:] != '.csv':
            continue
        category = '_'.join(cosine_sim_matrix.split('_')[2:])[:-4]
        
        cosine_sim_df = pd.read_csv(f'{cosine_sim_dir}{cosine_sim_matrix}', header=None)
        cosine_sim_matrices[category] = cosine_sim_df
        
        filter_df = pd.read_csv(f'{filter_table_dir}filter_table_{category}.csv', index_col=0)
        filter_matrices[category] = filter_df
        
        row_nums_df = pd.read_csv(f'{row_nums_dir}row_nums_{category}.csv', index_col=0)
        row_nums[category] = row_nums_df
        
        embeddings_df = pd.read_csv(f'{embeddings_dir}minilm-l6-v2_embeddings_{category}.csv', index_col=0)
        embeddings[category] = embeddings_df

    show_id_to_category = json.load(open('show_id_to_category.json'))
    show_id_to_name = json.load(open('show_id_to_name.json'))
    episode_id_to_name = json.load(open('episode_id_to_name.json'))
    
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    
    return (metadata_df,
            cosine_sim_matrices,
            filter_matrices,
            row_nums,
            embeddings,
            show_id_to_category,
            show_id_to_name,
            episode_id_to_name,
            model
           )

metadata_df, cosine_sim_matrices, filter_matrices, row_nums, embeddings, show_id_to_category, show_id_to_name, episode_id_to_name, model = data_loading()

def get_episode_ids(row_num_df, idxs):
    episode_ids = []
    for idx in idxs:
        episode_ids.append(row_num_df['episode_id'].iloc[idx])
    return episode_ids

def get_top_similar_podcast_idx(category_cosine_sims, category_filter, n_similar):
    '''
    Gets indices of the top n similar podcasts. If there aren't n to return, it returns the max possible number
    '''
    ordered_cosine_sims = category_cosine_sims.iloc[row_num].sort_values(ascending=False)
    ordered_filt = category_filter.iloc[row_num]
    top_n = []
    for i in ordered_cosine_sims.index:
        if not ordered_filt.iloc[i]:
            top_n.append(i)
        if len(top_n) == n_similar:
            break 
    return top_n

st.markdown('# Welcome to our podcast recommender!')
st.markdown('by Aditya Chander, Ritika Khurana, Taylor Mahler and Yuchen Luo')
st.markdown('***')

mode = st.selectbox('How would you like to get podcast recommendations?',
                    options = ['Select option...','Enter episode description', 'Select an episode I\'ve already heard'])

if mode == 'Select an episode I\'ve already heard':

    st.markdown('What podcasts are you listening to? We\'ll recommend you some similar episodes from different shows.')
    show_id = st.selectbox('Select podcast',
                           options=sorted(show_id_to_name.keys(), key=lambda x: show_id_to_name[x].strip().lower()),
                           format_func=lambda x: show_id_to_name[x].strip())
    episode_ids = metadata_df['episode_filename_prefix'][metadata_df['show_filename_prefix']==show_id].values
    episode_id = st.selectbox('Select episode', options=sorted(episode_ids, key = lambda x: episode_id_to_name[x].strip().lower()),
                              format_func=lambda x: episode_id_to_name[x])
    n_similar = st.slider('How many recommendations would you like?', min_value=1, max_value=20, value=5, step=1)

    # st.write(f'{show_id_to_category[show_id]}, {show_id}, {episode_id}')

    category_cosine_sims = cosine_sim_matrices[show_id_to_category[show_id]].reset_index(drop=True)
    category_filter = filter_matrices[show_id_to_category[show_id]].reset_index(drop=True)
    category_filter.columns = range(len(category_filter.columns))
    category_row_nums = row_nums[show_id_to_category[show_id]]

    row_num = category_row_nums[category_row_nums['show_id']==show_id][category_row_nums['episode_id']==episode_id].index[0]
    # st.write(row_num)

    similar_episode_ids = get_episode_ids(category_row_nums,
                                          get_top_similar_podcast_idx(category_cosine_sims, category_filter, n_similar))
    # similar_show_and_epis = category_row_nums[category_row_nums['episode_id'].isin(similar_episode_ids)]
    similar_show_and_epis = [[],[]]
    for episode in similar_episode_ids:
        similar_show_and_epis[0].append(category_row_nums[category_row_nums['episode_id']==episode]['show_id'].iloc[0])
        similar_show_and_epis[1].append(category_row_nums[category_row_nums['episode_id']==episode]['episode_id'].iloc[0])
    similar_show_and_epis = pd.DataFrame(data={'show_id':similar_show_and_epis[0], 'episode_id':similar_show_and_epis[1]})

    if len(similar_show_and_epis) < n_similar:
        st.markdown('Sorry, there were not enough episodes within this category that weren\'t from the same podcast.')

    st.markdown('**You may also like:**')

    similar_show_epi_names_df = pd.DataFrame(data={
        'Show name': [show_id_to_name[similar_show_and_epis['show_id'].iloc[i]] for i in range(len(similar_show_and_epis))],
        'Episode name': [episode_id_to_name[similar_show_and_epis['episode_id'].iloc[i]] for i in range(len(similar_show_and_epis))]})

    styler = similar_show_epi_names_df.style.hide_index()
    st.write(styler.to_html(), unsafe_allow_html=True)
    
elif mode == 'Enter episode description':
    
    episode_description = st.text_input('Enter episode description', max_chars=256)
    n_similar = st.slider('How many recommendations would you like?', min_value=1, max_value=20, value=5, step=1)
    
    if episode_description != '':
        embedding_desc = model.encode(episode_description).reshape(1,-1)
        curr_idx = 0
        cosine_sims_desc = np.zeros(len(metadata_df))
        shows = []
        episodes = []
        for category in sorted(embeddings.keys()):
            # st.write(category)
            shows.extend(list(embeddings[category]['show_id'].values))
            episodes.extend(list(embeddings[category]['episode_id'].values))
            tmp_embeddings = embeddings[category].values[:,3:]
            cosine_sims_desc[curr_idx:curr_idx+tmp_embeddings.shape[0]] = cosine_similarity(embedding_desc, tmp_embeddings)
            curr_idx += tmp_embeddings.shape[0]
        sim_episode_idx = np.argsort(cosine_sims_desc)[-1:-n_similar-1:-1]
        
        similar_show_and_epis = pd.DataFrame(data={'show_id':[shows[i] for i in sim_episode_idx],
                                                   'episode_id':[episodes[i] for i in sim_episode_idx]})
        if len(similar_show_and_epis) < n_similar:
            st.markdown('Sorry, there were not enough episodes within this category that weren\'t from the same podcast.')

        st.markdown('**You may also like:**')

        similar_show_epi_names_df = pd.DataFrame(data={
            'Show name': [show_id_to_name[similar_show_and_epis['show_id'].iloc[i]] for i in range(len(similar_show_and_epis))],
            'Episode name': [episode_id_to_name[similar_show_and_epis['episode_id'].iloc[i]] for i in range(len(similar_show_and_epis))]})

        styler = similar_show_epi_names_df.style.hide_index()
        st.write(styler.to_html(), unsafe_allow_html=True)