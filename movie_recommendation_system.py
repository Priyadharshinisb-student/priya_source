import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample Data (Replace with your actual data)
movies = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Gajini', 'Vikram', 'Dada', 'Mersal', 'Comali'],
    'genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy']
}

movies_df = pd.DataFrame(movies)

ratings = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 5],
    'rating': [5, 4, 3, 5, 4, 2, 1, 5, 3]
}

ratings_df = pd.DataFrame(ratings)

# Merge data
movie_ratings = pd.merge(ratings_df, movies_df, on='movie_id')

# Calculate item-based similarity (cosine similarity)
movie_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
movie_similarity = cosine_similarity(movie_matrix)

# Function to get recommendations
def get_recommendations(user_id, movie_similarity, movie_matrix):
    user_ratings = movie_matrix.loc[user_id].values.reshape(1, -1)  # Get user's ratings
    similarity_scores = cosine_similarity(user_ratings, movie_matrix)  # Calculate similarity with all movies
    
    # Sort recommendations based on similarity
    sorted_indices = similarity_scores.argsort()[0][::-1]
    
    # Exclude movies already rated by the user
    user_rated_movies = movie_matrix.columns[movie_matrix.loc[user_id] > 0].tolist()
    unrated_movies = [movie_matrix.columns[i] for i in sorted_indices if movie_matrix.columns[i] not in user_rated_movies][:5]
    
    return unrated_movies

# Get recommendations for user with ID 1
user_id = 1
recommendations = get_recommendations(user_id, movie_similarity, movie_matrix)

# Print recommendations
print(f"Recommendations for user {user_id}: {recommendations}")
