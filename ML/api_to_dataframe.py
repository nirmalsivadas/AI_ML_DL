
import pandas as pd
import requests

# try:
#     response = requests.get(
#         "https://api.themoviedb.org/3/movie/top_rated?api_key=YOUR_API_KEY&language=en-US&page=1",
#         timeout=10
#     )
#     response.raise_for_status()
#     data = response.json()
#     temp_df = pd.DataFrame(data["results"])[["id", "title", "overview", "release_date", "popularity", "vote_average", "vote_count"]]
#     print(temp_df.head())
# except requests.exceptions.RequestException as e:
#     print("API request failed:", e)

df = pd.DataFrame()

for i in range(1,429):
    response = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page={}'.format(i))
    temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
    df = df.append(temp_df,ignore_index=True)

print(df.head())

df.to_csv('files/movies.csv')