 import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load the movie dataset
def load_data():
    return pd.read_csv("movies.csv")

# Main function
def main():
    # Load data
    movies_df = load_data()

    # Load Surprise dataset
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(movies_df[['userId', 'movieId', 'rating']], reader)

    # Split the data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Use SVD algorithm
    algo = SVD()

    # Train the algorithm on the trainset
    algo.fit(trainset)

    # Make predictions on the testset
    predictions = algo.test(testset)

    # Example: Get top recommendations for a user
    user_id = 1
    user_movies = movies_df[movies_df['userId'] == user_id]['movieId'].unique()
    user_unseen_movies = movies_df[~movies_df['movieId'].isin(user_movies)]

    top_n = 10
    recommendations = {}
    for movie_id in user_unseen_movies['movieId']:
        predicted_rating = algo.predict(user_id, movie_id).est
        recommendations[movie_id] = predicted_rating

    top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print("Top Recommendations:")
    for movie_id, rating in top_recommendations:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        print(f"- {movie_title} (Predicted Rating: {rating:.2f})")

if __name__ == "__main__":
    main()