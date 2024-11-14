#Import Librabry
from implicit.als import AlternatingLeastSquares
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os

#read modified csv file

directory='C:/Users/HP/Documents/AIML_Project/Song_Recommendation_System/Data'
File_name='modified_data.csv'
Train_csv_path=os.path.join(directory,File_name)
data = pd.read_csv(Train_csv_path, encoding='ISO-8859-1')

# Create mapping for user_id and song_id
user_ids = data['user_id'].unique().tolist()
song_ids = data['song_id'].unique().tolist()

# Mapping user_id and song_id to index
user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
song_to_idx = {song: idx for idx, song in enumerate(song_ids)}

# Apply mapping to the train data
data['user_idx'] = data['user_id'].map(user_to_idx)
data['song_idx'] = data['song_id'].map(song_to_idx)


# Convert to sparse matrix format
train_sparse = coo_matrix((count, (rows, cols)), shape=(len(user_ids), len(song_ids)))

# Initialize & Train ALS model
als_model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)

als_model.fit(train_sparse)




#Reading Recommendation_Val file
directory='C:/Users/HP/Documents/AIML_Project/Song_Recommendation_System/Data'
Val_File_Name='Recommendation_Val.csv'
Val_File_Path=os.path.join(directory, Val_File_Name)
recommendation_df = pd.read_csv(Val_File_Path)


def get_recommendations(user_id, num_recommendations=10):
    # Convert the sparse matrix to CSR format for efficient row slicing
    train_sparse_csr = train_sparse.tocsr()

    # Get the user index from the user_id mapping
    user_idx = user_to_idx[user_id]

    # Use the ALS model to recommend songs for the user
    user_recommendations = als_model.recommend(user_idx, train_sparse_csr[user_idx], N=num_recommendations, filter_already_liked_items=True)

    # Extracting song IDs from the recommendations
    recommended_song_ids = [song_ids[i] for i in user_recommendations[0]]
    
    return recommended_song_ids


#Reading Recommendation_Val CSV file
Val_File_Name='Recommendation_Val.csv'
Val_File_Path=os.path.join(directory, Val_File_Name)
recommendation_df = pd.read_csv(Val_File_Path)

#Remove unnamed column
recommendation_df.drop(recommendation_df.columns[0],axis=1,inplace=True)

new_row= {recommendation_df.columns[0]: recommendation_df.columns[0]}
recommendation_df.loc[len(recommendation_df)]= new_row

# Preparing song recommendations for each user in recommendation_val file
recommendations = []
for user in recommendation_df['43683da3c6c5a93c7938ff550faf0d039a9a639a']:
    recommended_songs = get_recommendations(user)
    recommendations.append([user] + recommended_songs)

# Create a DataFrame with the recommended songs
recommendation_output = pd.DataFrame(recommendations, columns=['user_id'] + [f'recommended_song_{i+1}' for i in range(10)])

# Save the recommendations to a CSV file
recommendation_output.to_csv('final_recommendations.csv', index=False)