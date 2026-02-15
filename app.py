import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Movie Finder", page_icon="🎬", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('movies_clean.csv')
    df['genres'] = df['genres'].apply(ast.literal_eval)
    return df

@st.cache_resource
def load_models():
    with open('tfidf_model.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('genre_list.pkl', 'rb') as f:
        genre_list = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return tfidf, tfidf_matrix, genre_list, scaler

# Load everything
try:
    movies_df = load_data()
    tfidf, tfidf_matrix, genre_list, scaler = load_models()
except FileNotFoundError as e:
    st.error("Required files not found!")
    st.info("Please run: python prepare_data.py and python train.py first")
    st.stop()

# Title
st.title("Smart Movie Recommender")
st.markdown("Find movies that match your preferences!")

# Sidebar filters
st.sidebar.header("Your Preferences")

# Genre selection
all_genres = sorted(set(g for genres in movies_df['genres'] for g in genres))
selected_genres = st.sidebar.multiselect("Select Genres", all_genres)

# Rating
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 6.0, 0.5)

# Year range
min_year = int(movies_df['year'].min())
max_year = int(movies_df['year'].max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))

# Runtime
runtime_range = st.sidebar.slider("Duration (minutes)", 0, 300, (60, 180), 10)

# Language
languages = sorted(movies_df['language_name'].unique())
selected_lang = st.sidebar.multiselect("Language", languages, default=['English'])

# Similar to
similar_to = st.sidebar.text_input("Similar to (optional)", placeholder="e.g., Inception")

# Number of results
num_results = st.sidebar.slider("Number of results", 5, 20, 10)

# Search button
if st.sidebar.button("🔍 Find Movies", type="primary"):
    
    # Filter movies
    filtered = movies_df.copy()
    
    # Apply filters
    if selected_genres:
        filtered = filtered[filtered['genres'].apply(lambda x: any(g in x for g in selected_genres))]
    
    filtered = filtered[filtered['rating'] >= min_rating]
    filtered = filtered[(filtered['year'] >= year_range[0]) & (filtered['year'] <= year_range[1])]
    filtered = filtered[(filtered['runtime'] >= runtime_range[0]) & (filtered['runtime'] <= runtime_range[1])]
    
    if selected_lang:
        filtered = filtered[filtered['language_name'].isin(selected_lang)]
    
    if len(filtered) == 0:
        st.warning("No movies match your criteria. Try relaxing filters!")
    else:
        st.success(f"Found {len(filtered)} movies!")
        
        # Rank by similarity if "similar to" is provided
        if similar_to:
            # Find reference movie
            ref = movies_df[movies_df['title'].str.contains(similar_to, case=False, na=False)]
            
            if not ref.empty:
                ref_idx = ref.index[0]
                st.info(f"📌 Finding movies similar to: **{ref.iloc[0]['title']}**")
                
                # Calculate similarity
                filtered_indices = filtered.index.tolist()
                ref_vector = tfidf_matrix[ref_idx]
                filtered_vectors = tfidf_matrix[filtered_indices]
                
                similarities = cosine_similarity(filtered_vectors, ref_vector).flatten()
                filtered = filtered.copy()
                filtered['similarity'] = similarities
                
                # Combine with rating
                filtered['score'] = 0.6 * (filtered['rating'] / 10.0) + 0.4 * filtered['similarity']
            else:
                st.warning(f"Couldn't find '{similar_to}'. Showing by rating instead.")
                filtered['score'] = filtered['rating'] / 10.0
        else:
            # Rank by rating and popularity
            filtered['score'] = (
                0.7 * (filtered['rating'] / 10.0) + 
                0.3 * (filtered['popularity'] / filtered['popularity'].max())
            )
        
        # Get top results
        top_movies = filtered.nlargest(num_results, 'score')
        
        # Display results
        st.markdown("---")
        st.subheader(f"🎯 Top {len(top_movies)} Recommendations")
        
        for idx, (_, movie) in enumerate(top_movies.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"### {idx}. {movie['title']} ({int(movie['year'])})")
                    
                    # Metadata
                    st.markdown(
                        f"⭐ **{movie['rating']:.1f}/10** "
                        f"({int(movie['vote_count']):,} votes) | "
                        f"⏱️ {int(movie['runtime'])} min | "
                        f"🗣️ {movie['language_name']}"
                    )
                    
                    # Genres
                    genres_str = ", ".join(movie['genres'][:3])
                    st.markdown(f"**Genres:** {genres_str}")
                    
                    # Overview
                    overview = movie['overview']
                    if len(overview) > 250:
                        overview = overview[:250] + "..."
                    st.markdown(f"**Plot:** {overview}")
                
                with col2:
                    score_pct = int(movie['score'] * 100)
                    st.metric("Match", f"{score_pct}%")
                
                # Explanation
                reasons = []
                if selected_genres:
                    matched = [g for g in movie['genres'] if g in selected_genres]
                    if matched:
                        reasons.append(f"Matches {', '.join(matched)}")
                
                if movie['rating'] >= 8.0:
                    reasons.append(f"Highly rated ({movie['rating']:.1f})")
                
                if similar_to and 'similarity' in movie and movie['similarity'] > 0.6:
                    reasons.append(f"Similar to {similar_to}")
                
                if reasons:
                    st.info(f"💡 {' • '.join(reasons)}")
                
                st.markdown("---")

else:
    # Welcome screen
    st.info("👈 Set your preferences in the sidebar and click 'Find Movies'")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", len(movies_df))
    col2.metric("Genres", len(all_genres))
    col3.metric("Years", f"{min_year}-{max_year}")
    col4.metric("Languages", len(languages))
    
    # Top rated
    st.markdown("### 🌟 Top Rated Movies")
    top_rated = movies_df.nlargest(5, 'rating')[['title', 'year', 'rating', 'genres']]
    for _, m in top_rated.iterrows():
        st.markdown(f"**{m['title']}** ({int(m['year'])}) - ⭐ {m['rating']:.1f}")