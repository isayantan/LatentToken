import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_text

# Download a file from a URL and save it to ../data/ folder
def download_file(url, file_name):
    # Define the save path
    save_folder = "../data/"
    save_path = os.path.join(save_folder, file_name)
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Send HTTP request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the content to a file
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"File successfully downloaded and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        
        
# Load a dataset from a file
def load_dataset(dataset, url=None):
    file_path = os.path.join("../data/", dataset)
    if dataset == "AirlineTweets.csv":
        url = "https://lazyprogrammer.me/course_files/AirlineTweets.csv"

    if not os.path.exists(file_path):
        download_file(url, dataset)
        
    return pd.read_csv(file_path)

# Load X and y
def prepare_data(dataset, test_size=0.2, random_state=42):
    # Load dataset
    if dataset == "AirlineTweets.csv":
        df = load_dataset(dataset)
            
        # Split into X and y and preprocess
        X = df["text"].map(preprocess_text)
        
        target_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        y = df["airline_sentiment"].map(target_map)
        
        # Split into X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= random_state)
    return X_train, X_test, y_train, y_test