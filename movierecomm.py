import base64

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="MovieMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_data():
    return pd.read_csv("movies.csv").copy()


def calculate_similarity(movie_genres_1, movie_genres_2):
    if not movie_genres_1 or not movie_genres_2:
        return 0.0
    genres_1 = set(str(movie_genres_1).split("|"))
    genres_2 = set(str(movie_genres_2).split("|"))
    genres_1.discard("nan")
    genres_2.discard("nan")
    if not genres_1 or not genres_2:
        return 0.0
    intersection = genres_1.intersection(genres_2)
    return len(intersection) / len(genres_1.union(genres_2))


def get_recommendations(movie_title, movies_df, threshold=0.2, limit=12):
    movie_rows = movies_df[movies_df["title"] == movie_title]
    if movie_rows.empty:
        return []

    movie_genres = movie_rows.iloc[0]["genres"]
    recommendations = []

    for _, row in movies_df.iterrows():
        if row["title"] == movie_title:
            continue
        similarity = calculate_similarity(movie_genres, row["genres"])
        if similarity >= threshold:
            recommendations.append(
                {
                    "title": row["title"],
                    "genres": str(row["genres"]),
                    "score": similarity,
                }
            )

    return sorted(recommendations, key=lambda item: item["score"], reverse=True)[:limit]


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as file:
        return base64.b64encode(file.read()).decode()


def set_background_image(path):
    encoded = get_base64_of_bin_file(path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(8, 10, 18, 0.78), rgba(8, 10, 18, 0.94)),
                              url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .block-container {{
            max-width: 1100px;
            padding-top: 3rem;
            padding-bottom: 4rem;
        }}
        .hero {{
            text-align: center;
            padding: 2rem 1rem 1.5rem;
        }}
        .hero h1 {{
            font-size: clamp(2.4rem, 6vw, 4.5rem);
            margin-bottom: .35rem;
            letter-spacing: -0.04em;
        }}
        .hero p {{
            color: #b7bdca;
            font-size: 1.1rem;
            margin: 0 auto;
            max-width: 650px;
        }}
        .movie-card {{
            padding: 1.25rem;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 16px;
            background: rgba(18, 21, 31, .78);
            min-height: 145px;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }}
        .movie-title {{
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: .65rem;
        }}
        .genre {{
            display: inline-block;
            padding: .22rem .55rem;
            margin: .15rem .2rem .15rem 0;
            border-radius: 999px;
            background: rgba(255,255,255,.08);
            color: #cbd0db;
            font-size: .76rem;
        }}
        .score {{
            color: #9ca3af;
            font-size: .78rem;
            margin-top: .7rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    movies_df = load_data()
    set_background_image("background.jpg")

    st.markdown(
        """
        <div class="hero">
            <h1>🎬 MovieMatch</h1>
            <p>Pick a movie you love and discover films with a similar vibe.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    movies = sorted(movies_df["title"].dropna().unique().tolist())
    selected_movie = st.selectbox(
        "What are you in the mood for?",
        movies,
        index=None,
        placeholder="Search for a movie...",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"{len(movies):,} movies available")
    with col2:
        find_movies = st.button("✨ Find movies", type="primary", use_container_width=True)

    if find_movies:
        if not selected_movie:
            st.warning("Choose a movie first, then we'll find your matches.")
            return

        with st.spinner("Finding movies with a similar vibe..."):
            recommendations = get_recommendations(selected_movie, movies_df)

        st.markdown(f"### Movies like **{selected_movie}**")

        if not recommendations:
            st.info("No close matches found. Try another movie with more genres.")
            return

        st.caption(f"Top {len(recommendations)} matches, ranked by genre similarity")
        columns = st.columns(3)

        for index, movie in enumerate(recommendations):
            with columns[index % 3]:
                genres = [genre for genre in movie["genres"].split("|") if genre and genre != "nan"]
                genre_html = "".join(f'<span class="genre">{genre}</span>' for genre in genres[:4])
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div>{genre_html}</div>
                        <div class="score">{movie['score']:.0%} genre match</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
