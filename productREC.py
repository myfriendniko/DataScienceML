import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarit
# ==============================================================================
# Simulate a synthetic dataset
# ==============================================================================
print("Generating synthetic user-product ratings data...")

# Define users and products
users = ['User_A', 'User_B', 'User_C', 'User_D', 'User_E']
products = ['Product_1', 'Product_2', 'Product_3', 'Product_4', 'Product_5', 'Product_6']

# Create a dictionary of ratings. NaN = UNRATEDPRODUCT.
data = {
    'Product_1': [5, 4, 0, 0, 3],
    'Product_2': [4, 5, 5, 0, 0],
    'Product_3': [0, 4, 5, 5, 0],
    'Product_4': [0, 0, 4, 5, 5],
    'Product_5': [5, 0, 0, 4, 5],
    'Product_6': [3, 0, 0, 0, 4]
}
ratings_df = pd.DataFrame(data, index=users)
ratings_df.replace(0, np.nan, inplace=True) # Replace 0 with NaN for "not rated"

print("\nOriginal Ratings DataFrame:")
print(ratings_df)

# ==============================================================================
# Step 3: Calculate User-to-User Similarity
# ==============================================================================
print("\nCalculating user similarity matrix...")

# Fill NaNs with 0 for the similarity calculation
ratings_matrix = ratings_df.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(ratings_matrix)

# Create a DataFrame for the similarity matrix for better readability
user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)

print("\nUser Similarity Matrix:")
