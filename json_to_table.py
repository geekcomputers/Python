//uncomment below line to test the code
//jsondata = '{"0001":{"FirstName":"John","LastName":"Mark","MiddleName":"Lewis","username":"johnlewis2","password":"2910"}}'

//Used panndas dataframe to convert json data to dataframes table
import json
import pandas as pd
jdata = json.loads(jsondata)
df = pd.DataFrame(jdata)
print(df.T)
