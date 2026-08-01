import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a polished look
def apply_custom_styling():
    st.markdown("""
    <style>
    /* Main background with gradient overlay */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar / main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title styling */
    h1 {
        color: #f5f5f5 !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        text-align: center;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        margin-bottom: 0.2rem !important;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #aaa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Card containers */
    .movie-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: rgba(255,255,255,0.12);
    }
    
    /* Movie title inside card */
    .movie-card .title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #ffffff;
    }
    .movie-card .meta {
        font-size: 0.85rem;
        color: #b0b0b0;
        margin-top: 4px;
    }
    .movie-card .meta span {
        margin-right: 12px;
    }
    .movie-card .similarity-badge {
        display: inline-block;
        background: #6c63ff;
        color: #fff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-top: 6px;
    }
    
    /* Selected movie highlight */
    .selected-info {
        background: rgba(108, 99, 255, 0.15);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .selected-info .label {
        color: #999;
        font-size: 0.85rem;
    }
    .selected-info .value {
        color: #f0f0f0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #4834d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(108, 99, 255, 0.4);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #ccc !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #f0f0f0 !important;
        border-radius: 8px !important;
    }
    .stSelectbox > div > div:focus {
        border-color: #6c63ff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        color: #ccc !important;
        background: rgba(255,255,255,0.05);
        border-radius: 8px !important;
    }
    
    /* Progress / spinner */
    .stSpinner > div {
        border-color: #6c63ff !important;
    }
    
    /* Metrics */
    .recommendation-count {
        text-align: center;
        color: #6c63ff;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 0.5rem 0 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ---------- Data helpers ----------

@st.cache_data(show_spinner="Loading movie data...")
def load_data():
    return pd.read_csv("movies.csv").copy()


@st.cache_data(show_spinner="Computing similarity matrix...")
def compute_similarity_matrix(data):
    try:
        genres_list = data['genres'].tolist()
        vectorizer = CountVectorizer(binary=True)
        genre_matrix = vectorizer.fit_transform(genres_list)
        similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)
        return similarity_matrix
    except Exception as e:
        st.error(f"Could not compute similarity matrix: {e}")
        return None


def calculate_similarity(movie_genres_1, movie_genres_2):
    if not movie_genres_1 or not movie_genres_2:
        return 0.0
    try:
        genres_1 = set(movie_genres_1.split('|'))
        genres_2 = set(movie_genres_2.split('|'))
    except AttributeError:
        return 0.0
    intersection = genres_1.intersection(genres_2)
    if not intersection:
        return 0.0
    similarity = len(intersection) / (len(genres_1) + len(genres_2) - len(intersection))
    return similarity


def get_recommendations(movie_title, movies_df, similarity_matrix, threshold=0.2):
    movie_row = movies_df[movies_df['title'] == movie_title]
    movie_genres = movie_row['genres'].values[0]
    recommendations = []
    for index, row in movies_df.iterrows():
        if row['title'] != movie_title:
            similarity = calculate_similarity(movie_genres, row['genres'])
            if isinstance(similarity, (int, float)) and similarity >= threshold:
                recommendations.append((row['title'], similarity, row))
    # Sort by similarity descending
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations


def format_genres(genres_str):
    """Convert 'Action Adventure Fantasy' -> 'Action, Adventure, Fantasy'"""
    if not isinstance(genres_str, str):
        return ""
    return genres_str.replace(' ', ' · ').replace('|', ' · ')


def extract_year(release_date):
    """Extract year from release date string."""
    if not isinstance(release_date, str):
        return ""
    match = re.search(r'(\d{4})', release_date)
    return match.group(1) if match else ""


def format_runtime(minutes):
    try:
        mins = float(minutes)
        h = int(mins // 60)
        m = int(mins % 60)
        if h > 0:
            return f"{h}h {m}m"
        return f"{m}m"
    except (ValueError, TypeError):
        return ""


# ---------- Main ----------

def main():
    apply_custom_styling()

    # Load data
    movies_df = load_data()
    similarity_matrix = compute_similarity_matrix(movies_df)

    # Title section
    st.markdown("<h1>🎬 Movie Recommender</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Pick a movie you love, and discover similar ones based on genre</p>',
        unsafe_allow_html=True,
    )

    # Layout: movie selection + button in a compact row
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected_movie = st.selectbox(
            "Choose a movie",
            movies_df['title'].values,
            placeholder="Search or select a movie...",
        )
    with col2:
        st.write("")  # vertical spacer
        st.write("")
        get_recs = st.button("🎯 Get Recommendations", use_container_width=True)
    with col3:
        st.write("")
        st.write("")
        # Similarity threshold slider (hidden behind expander for cleanliness)
        threshold = 0.2

    with st.expander("⚙️ Adjust similarity threshold", expanded=False):
        threshold = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Higher values return only closely related genres; lower values return broader suggestions.",
        )

    # Show selected movie info
    if selected_movie:
        sel_row = movies_df[movies_df['title'] == selected_movie].iloc[0]
        genres_str = format_genres(sel_row.get('genres', ''))
        year_str = extract_year(sel_row.get('release_date', ''))
        rating = sel_row.get('vote_average', '')
        runtime_str = format_runtime(sel_row.get('runtime', ''))

        parts = []
        if year_str:
            parts.append(f"📅 {year_str}")
        if genres_str:
            parts.append(f"🏷️ {genres_str}")
        if runtime_str:
            parts.append(f"⏱ {runtime_str}")
        if rating:
            parts.append(f"⭐ {rating}/10")

        st.markdown(
            f"""
            <div class="selected-info">
                <div class="label">YOU SELECTED</div>
                <div class="value">{selected_movie}</div>
                <div style="color:#b0b0b0;font-size:0.9rem;margin-top:6px">{' &nbsp;·&nbsp; '.join(parts)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Get & display recommendations
    if get_recs and selected_movie:
        with st.spinner("Finding similar movies..."):
            recommendations = get_recommendations(
                selected_movie, movies_df, similarity_matrix, threshold=threshold
            )

        if recommendations:
            count = len(recommendations)
            st.markdown(
                f'<div class="recommendation-count">🎯 Found {count} similar movie{"s" if count != 1 else ""}</div>',
                unsafe_allow_html=True,
            )

            # Show in two columns
            half = (count + 1) // 2
            left_col, right_col = st.columns(2)

            for i, (title, score, row) in enumerate(recommendations):
                genres_str = format_genres(row.get('genres', ''))
                year_str = extract_year(row.get('release_date', ''))
                rating = row.get('vote_average', '')
                runtime_str = format_runtime(row.get('runtime', ''))

                meta_parts = []
                if year_str:
                    meta_parts.append(f"📅 {year_str}")
                if rating:
                    meta_parts.append(f"⭐ {rating}")
                if runtime_str:
                    meta_parts.append(f"⏱ {runtime_str}")
                meta_text = " · ".join(meta_parts)

                pct = int(score * 100)
                card = f"""
                <div class="movie-card">
                    <div class="title">{title}</div>
                    <div class="meta">{genres_str}</div>
                    <div class="meta">{meta_text}</div>
                    <div class="similarity-badge">{pct}% match</div>
                </div>
                """

                col = left_col if i < half else right_col
                col.markdown(card, unsafe_allow_html=True)
        else:
            st.info(
                "No recommendations found for this movie. Try lowering the similarity threshold in ⚙️ above.",
                icon="ℹ️",
            )

    elif get_recs and not selected_movie:
        st.warning("Please select a movie first.", icon="⚠️")


if __name__ == "__main__":
    main()
