import json
import pandas as pd
# //jsondata is a collection of data in json format

# //Used panndas dataframe to convert json data to dataframes table
jdata = json.loads(jsondata)
df = pd.DataFrame(jdata)
# //print(df.T)
