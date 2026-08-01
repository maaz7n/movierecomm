from __future__ import annotations

import base64
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_bg_from_local(image_file: str) -> None:
    """Encode a local image to base64 and inject it as a fixed background."""
    with open(image_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    /* Semi-transparent overlay so text stays readable */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.65);
        z-index: -1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and lightly clean the movie dataset."""
    df = pd.read_csv("movies.csv")
    # Normalise text fields
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["genres"] = df["genres"].fillna("").astype(str).str.strip()
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    df["director"] = df["director"].fillna("").astype(str).str.strip()
    df["tagline"] = df["tagline"].fillna("").astype(str).str.strip()
    # Numeric safety
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0.0)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def compute_similarity_matrix(genres_series: pd.Series) -> np.ndarray | None:
    """Return a cosine-similarity matrix from the genre strings."""
    try:
        vectorizer = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
        genre_matrix = vectorizer.fit_transform(genres_series.tolist())
        return cosine_similarity(genre_matrix, genre_matrix)
    except Exception:
        return None


def get_recommendations(
    movie_title: str,
    movies_df: pd.DataFrame,
    similarity_matrix: np.ndarray | None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top-N most similar movies based on the pre-computed similarity matrix."""
    # Locate the selected movie
    match = movies_df[movies_df["title"] == movie_title]
    if match.empty:
        return pd.DataFrame()

    idx = match.index[0]
    row_pos = movies_df.index.get_loc(idx)

    if similarity_matrix is not None and row_pos < similarity_matrix.shape[0]:
        sim_scores = list(enumerate(similarity_matrix[row_pos]))
        # Exclude the movie itself
        sim_scores = [s for s in sim_scores if s[0] != row_pos]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in sim_scores[:top_n]]
    else:
        # Fallback to Jaccard-like similarity if matrix is unavailable
        selected_genres = set(match.iloc[0]["genres"].split("|"))
        scores: list[tuple[int, float]] = []
        for pos, (_, row) in enumerate(movies_df.iterrows()):
            if pos == row_pos:
                continue
            g = set(row["genres"].split("|")) if row["genres"] else set()
            union = selected_genres | g
            inter = selected_genres & g
            score = len(inter) / len(union) if union else 0.0
            scores.append((pos, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_n]]

    return movies_df.iloc[top_indices].copy()


def star_rating(rating: float) -> str:
    """Convert a 0-10 rating into a simple star string."""
    filled = int(round(rating / 2))
    empty = 5 - filled
    return "★" * filled + "☆" * empty


def highlight_match(text: str, query: str) -> str:
    """Case-insensitive highlight of a substring."""
    if not query:
        return text
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group()}</mark>", text)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    add_bg_from_local("background.jpg")

    df = load_data()
    sim_matrix = compute_similarity_matrix(df["genres"])

    # ---- Header ----
    st.title("🎬 Movie Recommendation System")
    st.markdown("<p style='font-size:1.1rem; color:#ccc;'>Discover your next favourite film based on genre similarity.</p>", unsafe_allow_html=True)
    st.divider()

    # ---- Sidebar controls ----
    st.sidebar.header("⚙️ Settings")
    top_n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=20, value=8)
    show_details = st.sidebar.checkbox("Show movie details", value=True)
    show_year = st.sidebar.checkbox("Show release year", value=True)
    show_director = st.sidebar.checkbox("Show director", value=True)

    # ---- Search / Select ----
    search_query = st.text_input("🔎 Search movies", placeholder="Type a title...", value="")

    titles = df["title"].tolist()
    if search_query:
        filtered = [t for t in titles if search_query.lower() in t.lower()]
        if not filtered:
            st.warning("No movies match your search.")
            return
        selected_movie = st.selectbox("Select a movie:", filtered)
    else:
        selected_movie = st.selectbox("Select a movie:", titles)

    if not selected_movie:
        st.info("Choose a movie to get started.")
        return

    # ---- Selected movie info ----
    movie_row = df[df["title"] == selected_movie].iloc[0]

    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Rating", f"{movie_row['vote_average']:.1f}/10")
            st.markdown(f"<div style='font-size:1.5rem; color:#f5c518;'>{star_rating(movie_row['vote_average'])}</div>", unsafe_allow_html=True)
            if show_year and movie_row["year"] > 0:
                st.caption(f"📅 {int(movie_row['year'])}")
            if show_director and movie_row["director"]:
                st.caption(f"🎬 {movie_row['director']}")
        with col2:
            genres = movie_row["genres"].replace("|", ", ") if movie_row["genres"] else "Unknown"
            st.markdown(f"**Genres:** {genres}")
            if show_details and movie_row["overview"]:
                st.markdown(f"*{movie_row['overview']}*")
            if movie_row["tagline"]:
                st.caption(f"\"{movie_row['tagline']}\"")

    st.divider()

    # ---- Recommendations ----
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            recs = get_recommendations(selected_movie, df, sim_matrix, top_n=top_n)

        if recs.empty:
            st.error("No recommendations found for this movie.")
            return

        st.subheader("Recommendations")

        # Render in a responsive grid using columns
        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    # Title with optional highlight
                    title_html = highlight_match(row["title"], search_query)
                    st.markdown(f"#### {title_html}", unsafe_allow_html=True)

                    # Rating
                    st.markdown(
                        f"<div style='font-size:1.1rem; color:#f5c518;'>{star_rating(row['vote_average'])} "
                        f"<span style='color:#888;font-size:0.9rem;'>({row['vote_average']:.1f})</span></div>",
                        unsafe_allow_html=True,
                    )

                    # Meta
                    meta_parts: list[str] = []
                    if show_year and row["year"] > 0:
                        meta_parts.append(f"📅 {int(row['year'])}")
                    if show_director and row["director"]:
                        meta_parts.append(f"🎬 {row['director']}")
                    if meta_parts:
                        st.caption("  ·  ".join(meta_parts))

                    # Genres
                    g = row["genres"].replace("|", ", ") if row["genres"] else "Unknown"
                    st.markdown(f"<span style='color:#bbb;font-size:0.85rem;'>🎭 {g}</span>", unsafe_allow_html=True)

                    # Overview
                    if show_details and row["overview"]:
                        short_overview = row["overview"][:180]
                        if len(row["overview"]) > 180:
                            short_overview += "…"
                        st.markdown(f"<p style='color:#ddd;font-size:0.85rem;margin-top:0.5rem;'>{short_overview}</p>", unsafe_allow_html=True)

        # Summary footer
        st.divider()
        st.caption(f"Showing {len(recs)} recommendation(s) based on genre similarity.")


if __name__ == "__main__":
    main()
