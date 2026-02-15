Simple Movie Recommender

A clean, simple movie discovery system with filtering and similarity search.

What You Need

1. Python 3.9+
2. Kaggle account (free)
3. 5 minutes of your time

Quick Setup

### Step 1: Download Dataset

Go to: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Download `tmdb_5000_movies.csv` and put it in this folder.

### Step 2: Install Packages

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data

```bash
python prepare_data.py
```

This creates `movies_clean.csv`

### Step 4: Train Model

```bash
python train.py
```
(takes ~30 seconds)

### Step 5: Run App

```bash
streamlit run app.py
```

Opens in browser at http://localhost:8501

How It Works

1. **Filter** - Select genre, rating, year, runtime, language
2. **Rank** - Uses TF-IDF similarity + ratings to rank results
3. **Results** - Shows top matches with explanations

Files

- `app.py` - Streamlit web app
- `train.py` - Creates TF-IDF model
- `prepare.py` - Cleans the dataset
- `requirements.txt` - Dependencies

Features

- Multi-criteria filtering
- TF-IDF similarity search
- "Similar to" movie search
- Clean, fast UI
- Explainable recommendations

Tech Stack

- **Streamlit** - Web interface
- **Pandas** - Data processing
- **Scikit-learn** - TF-IDF vectorization
- **NumPy** - Numerical operations
