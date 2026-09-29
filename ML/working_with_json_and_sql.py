import postgresql.connector 
import pandas as pd

pd.read_json('train.json') # reading json file from folder

pd.read_json('https://api.exchangerate-api.com/v4/latest/INR') # reading json file from url

conn = postgresql.connector.connect(host='localhost',user='postgres',password='1234',database='kisan_setu_db')

df = pd.read_sql_query("SELECT * FROM users",conn)



