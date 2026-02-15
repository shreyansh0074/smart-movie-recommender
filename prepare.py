import pandas as pd
import ast

print("Loading dataset...")
df = pd.read_csv('tmdb_5000_movies.csv')

print(f"Loaded {len(df)} movies")

# Parse JSON columns
def parse_genres(x):
    try:
        genres = ast.literal_eval(x)
        return [g['name'] for g in genres]
    except:
        return []

df['genres'] = df['genres'].apply(parse_genres)

# Extract year from release_date
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Rename and select columns
df = df.rename(columns={'vote_average': 'rating'})

# Keep only necessary columns
df = df[['id', 'title', 'overview', 'genres', 'rating', 
         'vote_count', 'popularity', 'runtime', 'year', 'original_language']]

# Drop missing values
df = df.dropna(subset=['overview', 'title', 'year', 'runtime'])
df = df[df['runtime'] > 0]
df = df[df['genres'].apply(len) > 0]

# Map language codes to full names
language_map = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'hi': 'Hindi',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'ar': 'Arabic',
    'tr': 'Turkish',
    'sv': 'Swedish',
    'th': 'Thai',
    'id': 'Indonesian',
    'el': 'Greek',
    'fa': 'Persian',
}

# Add language_name column
df['language_name'] = df['original_language'].map(language_map)
# For unmapped languages, capitalize the code
df['language_name'] = df['language_name'].fillna(df['original_language'].str.upper())

print(f"Cleaned dataset: {len(df)} movies")
# print(f"Languages: {sorted(df['language_name'].unique())}")

# Save
df.to_csv('movies_clean.csv', index=False)
print("Saved to movies_clean.csv")