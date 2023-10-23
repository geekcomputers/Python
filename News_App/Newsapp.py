import os
import solara as sr
import yfinance as yf


from patterns import Company_Name
from datetime import datetime as date,timedelta

srart_date = date.today()
end_date = date.today() + timedelta(days=1)


def News(symbol):
    get_Data = yf.Ticker(symbol)
    # msft.news

    #news section 
    try:
        NEWS = get_Data.news
        # sr.Markdown(f"{NEWS}")
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
# News(select)



company = list(Company_Name.keys())
v=sr.reactive(company[0])

@sr.component
def Page():
    with sr.Column() as main:
        with sr.Sidebar():
            sr.Markdown("## **stock Analysis**")
            # sr.SliderInt(label="Ideal for placing controls")
            # sr.header("**srock Analysis**")
            sr.Select("Select stock",value=v,values=company)

            select=Company_Name.get(v.value)


            # sr.Text(select_company)
        # sr.Info("I'm in the main content area, put your main content here")

        News(select)

        # sr.FigurePlotly(qs.plots.daily_returns(ITC,benchmark="US"))
    return main



# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

# if __name__=="__main__":
#     app.run(debug=False)