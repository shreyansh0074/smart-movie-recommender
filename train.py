import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle

print("Loading cleaned data")
df = pd.read_csv('movies_clean.csv')

# Parse genres back to list
import ast
df['genres'] = df['genres'].apply(ast.literal_eval)

print(f"Training on {len(df)} movies")

# 1. TF-IDF on plot summaries
print("1. Creating TF-IDF vectors from plot summaries")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")

# 2. Create genre features (one-hot encoding)
print("2. Creating genre features")
all_genres = sorted(set(g for genres in df['genres'] for g in genres))
# print(f"   Found {len(all_genres)} genres: {all_genres}")

genre_matrix = np.zeros((len(df), len(all_genres)))
for i, genres in enumerate(df['genres']):
    for genre in genres:
        if genre in all_genres:
            genre_matrix[i, all_genres.index(genre)] = 1

# 3. Normalize numerical features
print("3. Normalizing numerical features")
scaler = StandardScaler()
numerical_features = df[['rating', 'vote_count', 'popularity', 'runtime', 'year']].values
numerical_features_scaled = scaler.fit_transform(numerical_features)

# 4. Save everything
print("4. Saving models")

with open('tfidf_model.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

with open('genre_list.pkl', 'wb') as f:
    pickle.dump(all_genres, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


print("Training Complete")
