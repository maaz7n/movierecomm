import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to load the movie dataset
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")


# Function to compute similarity matrix based on genres
@st.cache_data
def compute_similarity_matrix(genres):
    # Create binary vectors representing presence/absence of genres
    vectorizer = CountVectorizer(binary=True)
    genre_matrix = vectorizer.fit_transform(genres)

    # Compute cosine similarity between genre vectors
    similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)
    return similarity_matrix


# Function to get movie recommendations using the precomputed similarity matrix
def get_recommendations(movie_title, movies_df, similarity_matrix, top_n=10):
    try:
        movie_idx = movies_df[movies_df["title"] == movie_title].index[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the movie itself (similarity == 1.0)
    sim_scores = [s for s in sim_scores if s[0] != movie_idx]

    recommendations = []
    for idx, score in sim_scores[:top_n]:
        recommendations.append(
            {
                "title": movies_df.iloc[idx]["title"],
                "score": round(score, 3),
                "genres": movies_df.iloc[idx]["genres"],
            }
        )
    return recommendations


# Function to set background image from URL
def set_background_image(url):
    page_bg_img = """
    <style>
    .stApp {
        background-image: url("%s");
        background-size: cover;
    }
    </style>
    """ % url
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Main function
def main():
    # Load data
    movies_df = load_data()

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(movies_df["genres"].tolist())

    # Set background image from URL
    background_image_url = (
        "https://raw.githubusercontent.com/maaz7n/movierecomm/main/background.jpg"
    )
    set_background_image(background_image_url)

    # Streamlit UI
    st.title("Movie Recommendation System")

    # Select a movie
    selected_movie = st.selectbox("Select a movie:", movies_df["title"].values)

    # Get recommendations
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(
            selected_movie, movies_df, similarity_matrix
        )
        if recommendations:
            st.write("### Recommendations")
            for rec in recommendations:
                st.write(
                    f"- **{rec['title']}**  _(similarity: {rec['score']})_"
                )
                st.caption(f"Genres: {rec['genres']}")
        else:
            st.write("No recommendations found for this movie.")


if __name__ == "__main__":
    main()
