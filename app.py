import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from nltk.stem.porter import PorterStemmer

# Load TMDB API key
TMDB_API_KEY = "b783d96acb639eaf1e1d8ef2aad208af" 

# Streamlit page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[['genres', 'movie_id', 'cast', 'overview', 'title', 'keywords', 'crew']]
    movies.dropna(inplace=True)

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert1(obj):
        return [i['name'] for i in ast.literal_eval(obj)[:3]]

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert1)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tag1'] = movies.genres + movies.cast + movies.crew
    movies['tag2'] = movies.overview + movies.keywords
    movies = movies[['movie_id', 'title', 'tag1', 'tag2']]
    movies['tag1'] = movies['tag1'].apply(lambda x: " ".join(x))
    movies['tag2'] = movies['tag2'].apply(lambda x: " ".join(x))
    movies['tag1'] = movies['tag1'].apply(lambda x: x.lower())
    movies['tag2'] = movies['tag2'].apply(lambda x: x.lower())

    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(i) for i in text.split()])

    movies['tag1'] = movies['tag1'].apply(stem)
    movies['tag2'] = movies['tag2'].apply(stem)

    return movies

movies = load_data()

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

vector1 = normalize(cv.fit_transform(movies['tag1']))
vector2 = normalize(tfidf.fit_transform(movies['tag2']))
vectors = hstack([vector1, vector2])
similarity = cosine_similarity(vectors)

def fetch_poster(title):
    """Get movie poster from TMDB API"""
    try:
        query = title.replace(" ", "%20")
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
        response = requests.get(url)
        data = response.json()
        poster_path = data['results'][0]['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except:
        return "https://via.placeholder.com/300x450?text=No+Poster"

def recommend(movie):
    if movie not in movies['title'].values:
        return [], []
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    titles = [movies.iloc[i[0]].title for i in movie_list]
    posters = [fetch_poster(title) for title in titles]
    return titles, posters

# UI Layout
movie_list = movies['title'].values
selected_movie = st.selectbox("🎥 Select a movie to get recommendations:", movie_list)

if st.button("🔍 Recommend"):
    names, posters = recommend(selected_movie)
    if names:
        st.markdown("### ✅ Top 5 Recommended Movies:")
        cols = st.columns(5)
        for idx in range(5):
            with cols[idx]:
                st.image(posters[idx], use_column_width=True)
                st.caption(names[idx])
    else:
        st.warning("Movie not found in the dataset.")
