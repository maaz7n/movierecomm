import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Function to load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

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
    background_image_url = "https://raw.githubusercontent.com/maaz7n/movierecomm/main/background.jpg"  # Replace with your URL
    set_background_image(background_image_url)

    # Streamlit UI
    st.title('Movie Recommendation System')

    # Load Surprise dataset
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(movies_df[['userId', 'movieId', 'rating']], reader)

    # Split the data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Use SVD algorithm
    algo = SVD()

    # Train the algorithm on the trainset
    algo.fit(trainset)

    # Select a user
    selected_user = st.selectbox('Select a user:', movies_df['userId'].unique())

    # Get top recommendations for the selected user
    top_n = 10
    recommendations = {}
    for movie_id in movies_df['movieId'].unique():
        predicted_rating = algo.predict(selected_user, movie_id).est
        recommendations[movie_id] = predicted_rating

    top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    st.write("### Top Recommendations:")
    for movie_id, rating in top_recommendations:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        st.write(f"- {movie_title} (Predicted Rating: {rating:.2f})")

if __name__ == "__main__":
    main()