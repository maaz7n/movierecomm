import os
import base64

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def set_background_from_file(path):
    """Inject a local background image as a base64 CSS background."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .block-container {{
            background: rgba(0, 0, 0, 0.72);
            border-radius: 16px;
            padding: 2rem 2.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    """Load and lightly clean the movie dataset."""
    df = pd.read_csv("movies.csv")
    # Ensure genres is a string and never NaN
    df["genres"] = df["genres"].fillna("").astype(str)
    return df


@st.cache_resource
def compute_similarity_matrix(genres_series):
    """
    Compute a cosine-similarity matrix using binary genre vectors.
    Genres in this dataset are space-separated (e.g. 'Action Adventure').
    """
    # Use token_pattern that keeps single-word genres intact.
    vectorizer = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
    genre_matrix = vectorizer.fit_transform(genres_series)
    similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)
    return similarity_matrix


def display_movie_card(row, key_prefix="movie"):
    """Render a compact info card for a single movie row."""
    title = row.get("title", "Unknown")
    genres = row.get("genres", "")
    rating = row.get("vote_average")
    runtime = row.get("runtime")
    director = row.get("director", "N/A")
    release = row.get("release_date", "")
    overview = str(row.get("overview", ""))

    st.markdown(f"#### {title}")
    st.caption(f"Genres: {genres}")

    cols = st.columns(4)
    cols[0].metric("Rating", f"{rating:.1f}" if pd.notna(rating) else "N/A")
    cols[1].metric(
        "Runtime",
        f"{int(runtime)} min" if pd.notna(runtime) else "N/A",
    )
    cols[2].metric("Director", str(director)[:22])
    cols[3].metric("Year", str(release)[:4] if pd.notna(release) else "N/A")

    if overview:
        st.markdown(f"> {overview}")


def main():
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Background image
    if os.path.exists("background.jpg"):
        set_background_from_file("background.jpg")

    # Load data
    with st.spinner("Loading movie database…"):
        movies_df = load_data()

    # Precompute similarity matrix once
    with st.spinner("Analysing genre similarities…"):
        similarity_matrix = compute_similarity_matrix(movies_df["genres"])

    # Sidebar controls & info
    with st.sidebar:
        st.title("🎬 Movie Recommender")
        st.caption("Content-based suggestions powered by genre overlap.")
        st.divider()

        st.metric("Movies in Database", len(movies_df))
        all_genres = set()
        for g in movies_df["genres"]:
            all_genres.update(g.split())
        st.metric("Distinct Genres", len(all_genres))

        st.divider()
        top_n = st.slider("Max Recommendations", 5, 50, 15)
        similarity_threshold = st.slider(
            "Minimum Similarity",
            0.0,
            1.0,
            0.10,
            0.05,
            help="Lower values return more results; higher values require closer genre matches.",
        )

        st.divider()
        st.markdown("**How it works**")
        st.info(
            "Select a movie you like. The app compares genre profiles with every other film "
            "using cosine similarity and shows the closest matches, ranked by relevance."
        )

    # Main content
    st.header("Find Your Next Favourite Movie")
    selected_movie = st.selectbox(
        "Choose a movie you enjoyed:",
        movies_df["title"].values,
        index=0,
    )

    if st.button("🎥  Get Recommendations", type="primary", use_container_width=True):
        # Locate the selected movie in the DataFrame
        match = movies_df[movies_df["title"] == selected_movie]
        if match.empty:
            st.error("Selected movie not found in the dataset.")
            return

        idx = match.index[0]
        movie_row = movies_df.loc[idx]

        # --- Selected movie showcase ---
        with st.container():
            st.subheader("🍿  Your Pick")
            display_movie_card(movie_row)
            st.divider()

        # --- Recommendations ---
        sim_scores = list(enumerate(similarity_matrix[idx]))
        # Exclude self
        sim_scores = [(i, float(score)) for i, score in sim_scores if i != idx]
        # Filter by threshold
        sim_scores = [(i, score) for i, score in sim_scores if score >= similarity_threshold]
        # Sort descending
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

        if not sim_scores:
            st.warning(
                "No recommendations found at the current threshold. Try lowering it in the sidebar."
            )
            return

        st.subheader(f"Top {len(sim_scores)} Recommendations")
        for rank, (rec_idx, score) in enumerate(sim_scores, start=1):
            rec = movies_df.iloc[rec_idx]
            title = rec.get("title", "Unknown")
            genres = rec.get("genres", "")
            rating = rec.get("vote_average")
            runtime = rec.get("runtime")
            director = rec.get("director", "N/A")
            overview = str(rec.get("overview", ""))

            with st.expander(f"#{rank}  {title}   (match {score:.0%})"):
                col_info, col_score = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**Genres:** {genres}")
                    st.markdown(f"**Director:** {director}")
                    meta = f"**Rating:** {rating:.1f}" if pd.notna(rating) else "**Rating:** N/A"
                    if pd.notna(runtime):
                        meta += f"  ·  **Runtime:** {int(runtime)} min"
                    st.markdown(meta)
                    if overview:
                        st.markdown(f"> {overview}")
                with col_score:
                    st.progress(score, text=f"{score:.0%} match")

    # Footer
    st.divider()
    st.caption("Powered by Streamlit  ·  Data from TMDB")


if __name__ == "__main__":
    main()
