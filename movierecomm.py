import streamlit as st
import pandas as pd
import base64

# Function to load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

# Function to calculate similarity based on genres
def calculate_similarity(movie_genres_1, movie_genres_2):
    if not movie_genres_1 or not movie_genres_2:
        return 0  # If any of the genres is empty, return 0 similarity
    try:
        genres_1 = set(movie_genres_1.split('|'))
        genres_2 = set(movie_genres_2.split('|'))
    except AttributeError:
        return 0  # If genres are not in the expected format, return 0 similarity
    intersection = genres_1.intersection(genres_2)
    similarity = len(intersection) / (len(genres_1) + len(genres_2) - len(intersection))
    return similarity

# Function to get movie recommendations
def get_recommendations(movie_title, movies_df, threshold=0.2):
    movie_row = movies_df[movies_df['title'] == movie_title]
    movie_genres = movie_row['genres'].values[0]
    recommendations = []
    for index, row in movies_df.iterrows():
        if row['title'] != movie_title:
            similarity = calculate_similarity(movie_genres, row['genres'])
            if similarity >= threshold:
                recommendations.append(row['title'])
    return recommendations

# Function to convert image to base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image from URL
def set_background_image(url):
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("%s");
        background-size: cover;
    }
    </style>
    ''' % url
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Main function
def main():
    # Load data
    movies_df = load_data()

    # Set background image from URL
    background_image_url = "https://example.com/background.jpg"  # Replace with your URL
    set_background_image(background_image_url)

    # Streamlit UI
    st.title('Movie Recommendation System')

    # Select a movie
    selected_movie = st.selectbox('Select a movie:', movies_df['title'].values)

    # Get recommendations
    if st.button('Get Recommendations'):
        recommendations = get_recommendations(selected_movie, movies_df)
        if recommendations:
            st.write("### Recommendations")
            for movie in recommendations:
                st.write(f"- {movie}")
        else:
            st.write("No recommendations found for this movie.")

if __name__ == "__main__":
    main()
