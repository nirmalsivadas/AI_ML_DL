import pandas as pd
import requests
from io import StringIO

df = pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv') # opening the csv file from folder

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0"}
req = requests.get(url, headers=headers)
data = StringIO(req.text) 

pd.read_csv(data) # opening the csv file from url

pd.read_csv('movie_titles_metadata.tsv',sep='\t',names=['sno','name','release_year','rating','votes','genres'])

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',index_col='enrollee_id') # setting enrollee_id as index

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',header=1) # skipping the first row

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',usecols=['iq','cgpa']) # selecting specific columns from the dataset

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',usecols=['iq'],squeeze=True) # selecting specific columns from the dataset

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',nrows=100) # reading only first 100 rows

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',skiprows=range(1,100)) # skipping first 100 rows

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',encoding='latin-1') # reading csv file with latin-1 encoding to handle special characters

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv', sep=';', encoding="latin-1",error_bad_lines=False) # error_bad_lines=False to skip bad lines

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',dtype={'cgpa':int}).info() # converting target column to integer type

pd.read_csv('C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\ML\files\placement.csv',parse_dates=['iq']).info() # converting date column to datetime type


def rename(name):
    if name == "Royal Challengers Bangalore":
        return "RCB"
    else:
        return name
    
pd.read_csv('IPL Matches 2008-2020.csv',converters={'team1':rename}) # converting team1 column to custom function

pd.read_csv('aug_train.csv',na_values=['Male',]) # replacing Male with NaN

dfs = pd.read_csv('aug_train.csv',chunksize=5000) # reading csv file in chunks 

for chunks in dfs:
    print(chunks.shape)
