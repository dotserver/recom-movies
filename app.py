import pickle
import streamlit as st
import pandas as pd
import requests

# Load the dataset
df = pd.read_csv('netflix_titles.csv')
df_movies = df[df['type'] == 'Movie'].reset_index()
movies = pd.Series(df_movies.index, index=df_movies['title'])


# URL to download the classifier.pkl file from Google Drive
download_url = "https://drive.google.com/uc?id=1wQIAWfHsZLtlybocZP61rUZWY-rNOE-H"

# Function to download the file
def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)

# Download the classifier.pkl file
download_file(download_url, 'classifier.pkl')

# Load the trained model
try:
    with open('classifier.pkl', 'rb') as pickle_in:
        classifier = pickle.load(pickle_in)
except FileNotFoundError:
    st.error("The classifier.pkl file was not found. Make sure it is in the correct directory.")

def prediction_model(title):
    try:
        index = movies[title]
    except KeyError:
        return "error"
    
    similarity_scores = list(enumerate(classifier[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    movie_indices = [i[0] for i in similarity_scores]
    prediction = df_movies['title'].iloc[movie_indices].to_list()
    return prediction

def main():
    # st.title("MOVIE RECOMMENDATION")
    html_temp = """
    <div style="background-color:grey;padding:10px"><marquee>
    <h2 style="color:white;text-align:center;">Movie Recommendation</h2></marquee>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    title = st.selectbox("Select a Movie Title", df_movies.title.tolist())
    result = ""
    
    if st.button("Search"):
        result = prediction_model(title)
        if result == "error":
            movies_sample = df_movies.sample(n=5)
            results = movies_sample['title'].to_list()
            st.error("Oops, we couldn't find that movie. Check out these movies instead:")
        else:
            results = result
        
        for value in results:
            st.success(value)

if __name__ == '__main__':
    main()
