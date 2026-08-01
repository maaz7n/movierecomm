import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df["genres"] = df["genres"].fillna("").astype(str)
    return df


@st.cache_data
def compute_similarity_matrix(genres_series):
    genres_list = genres_series.tolist()
    vectorizer = CountVectorizer(binary=True, token_pattern=r"[^\s]+")
    genre_matrix = vectorizer.fit_transform(genres_list)
    return cosine_similarity(genre_matrix, genre_matrix)


def get_recommendations(movie_title, movies_df, sim_matrix, top_n=12):
    idx = movies_df.index[movies_df["title"] == movie_title].tolist()
    if not idx:
        return []
    movie_idx = idx[0]
    sim_scores = list(enumerate(sim_matrix[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != movie_idx]
    sim_scores = sim_scores[:top_n]
    rec_indices = [i[0] for i in sim_scores]
    recs = movies_df.iloc[rec_indices].copy()
    recs["similarity"] = [s[1] for s in sim_scores]
    return recs


def set_background_css(image_path: str):
    bg_css = """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    .movie-card {
        background: rgba(30, 30, 30, 0.75);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform .15s ease;
    }
    .movie-card:hover {
        transform: translateY(-3px);
    }
    .movie-title {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .movie-meta {
        font-size: 13px;
        color: #bbbbbb;
        margin-bottom: 8px;
    }
    .genre-chip {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 3px 10px;
        margin: 2px 4px 2px 0;
        font-size: 11px;
        color: #f0f0f0;
    }
    .match-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 700;
        color: #fff;
    }
    .stApp {
        background-image: url("data:image/jpeg;base64,%s");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stSelectbox label { color: #ffffff; font-weight: 600; }
    .stSlider label { color: #ffffff; font-weight: 600; }
    .css-1l40w4m { padding-left: 16px; padding-right: 16px; }
    </style>
    """ % image_path
    st.markdown(bg_css, unsafe_allow_html=True)


def file_to_base64(path: str) -> str:
    from base64 import b64encode
    with open(path, "rb") as f:
        return b64encode(f.read()).decode()


def main():
    bg_path = Path("background.jpg")
    bg_b64 = file_to_base64(str(bg_path)) if bg_path.exists() else ""
    set_background_css(bg_b64)

    st.markdown(
        '<h1 style="color:#ffffff; text-align:center; margin-top:0;">🎬 Movie Recommendation System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; color:#dddddd; margin-bottom:24px;">Pick a movie you like and discover similar ones based on genre overlap.</p>',
        unsafe_allow_html=True,
    )

    movies_df = load_data()
    sim_matrix = compute_similarity_matrix(movies_df["genres"])

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_movie = st.selectbox("Select a movie", movies_df["title"].values)
    with col2:
        top_n = st.slider("Number of recommendations", 3, 30, 12)

    selected_row = movies_df[movies_df["title"] == selected_movie].iloc[0]
    genres = selected_row["genres"].split()
    avg_rating = selected_row.get("vote_average", "N/A")
    year = str(selected_row.get("release_date", ""))[:4] if pd.notna(selected_row.get("release_date")) else "N/A"

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{selected_movie}</div>
            <div class="movie-meta">Year: {year} &nbsp;|&nbsp; Rating: {avg_rating} &nbsp;|&nbsp; Genres: {" ".join([f'<span class="genre-chip">{g}</span>' for g in genres])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recs = get_recommendations(selected_movie, movies_df, sim_matrix, top_n=top_n)
    if recs.empty:
        st.info("No recommendations found for this movie.")
        return

    st.markdown(f"<h3 style='color:#ffffff; margin-top:16px;'>Top {len(recs)} Recommendations</h3>", unsafe_allow_html=True)

    for _, row in recs.iterrows():
        title = row["title"]
        sim = row["similarity"]
        rating = row.get("vote_average", "N/A")
        yr = str(row.get("release_date", ""))[:4] if pd.notna(row.get("release_date")) else "N/A"
        rec_genres = str(row.get("genres", "")).split()
        pct = int(sim * 100)
        badge_color = "#e74c3c" if pct >= 80 else "#f39c12" if pct >= 50 else "#3498db"

        st.markdown(
            f"""
            <div class="movie-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <div class="movie-title">{title}</div>
                    <div class="match-badge" style="background:{badge_color};">{pct}% match</div>
                </div>
                <div class="movie-meta">Year: {yr} &nbsp;|&nbsp; Rating: {rating}</div>
                <div>{" ".join([f'<span class="genre-chip">{g}</span>' for g in rec_genres])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
