import os
import solara as sr
import yfinance as yf


from patterns import Company_Name
from datetime import datetime as date,timedelta

srart_date = date.today()
end_date = date.today() + timedelta(days=1)


def News(symbol):
    get_Data = yf.Ticker(symbol)
    
    #news section 
    try:
        NEWS = get_Data.news
        sr.Markdown(f"# News of {v.value} :")
        for i in range(len(NEWS)):
            sr.Markdown("\n********************************\n")
            sr.Markdown(f"## {i+1}.   {NEWS[i]['title']} \n ")
            sr.Markdown(f"**Publisher** : {NEWS[i]['publisher']}\n")
            sr.Markdown(f"**Link** : {NEWS[i]['link']}\n")
            sr.Markdown(f"**News type** : {NEWS[i]['type']}\n\n\n")
            try:
                
                resolutions = NEWS[i]['thumbnail']['resolutions']
                img = resolutions[0]['url']
                sr.Image(img)

            except:
                pass
    except Exception as e:
        sr.Markdown(e)
        sr.Markdown("No news available")




company = list(Company_Name.keys())
v=sr.reactive(company[0])

@sr.component
def Page():
    with sr.Column() as main:
        with sr.Sidebar():
            sr.Markdown("## **stock Analysis**")
            sr.Select("Select stock",value=v,values=company)

            select=Company_Name.get(v.value)


        News(select)

    return main

