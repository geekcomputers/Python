import glob
import csv
import pdb
import os
import pandas as pd 

def main():
    directory = []
    for dirs in os.walk("."):
        directory.append(dirs)
    folders = directory[0][1]
    for ff in folders:
        if ff != ".git":
            allFiles = glob.glob(ff + "/*.csv")
            frame = pd.DataFrame()
            dfs = []
            for files in allFiles:
                df = pd.read_csv(files,index_col=None, header=0)
                dfs.append(df)
                frame = pd.concat(dfs)
            frame.to_csv(ff+"/results.csv")
main()
