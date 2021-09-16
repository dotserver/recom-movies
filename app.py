 
import pickle
import streamlit as st
import pandas as pd


df = pd.read_csv('netflix_titles.csv')
df_movies = (df[df['type'] == 'Movie']).reset_index()
movies = pd.Series(df_movies.index,index=df_movies['title'])

print("loading pickle file")
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
print("loaded pickle file")

def prediction_model(title):
    
    try:
        index=movies[title]
    except KeyError:
        return "error"
    print("starting prediction")
    similarity_scores=list(enumerate(classifier[index]))
    similarity_scores=sorted(similarity_scores, key=lambda x:x[1], reverse=True)
    similarity_scores=similarity_scores[1:11]
    movie_indices = [i[0] for i in similarity_scores]
    prediction = df_movies['title'].iloc[movie_indices].to_list()
    print(prediction)
    return prediction


def main():
    st.title("MOVIE RECCOMENDATION")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Movie Recomendation</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    title = st.text_input("Title","Input title here")
    result=""
    if st.button("Search"):
        result=prediction_model(title)
    if result == "error":
        movies = df_movies.sample(n=5)
        results = movies['title'].to_list()
        st.error("Oops, we couldn't find that movie")
        st.write("Check out these movies instead:")
    else:
        results = result
    for value in results:
        st.success(value)
    
if __name__=='__main__':
    main()
    
    