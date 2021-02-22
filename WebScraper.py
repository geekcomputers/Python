import csv
from bs4 import BeautifulSoup
from msedge.selenium_tools import Edge, EdgeOptions

def get_url(search_term):
    generic = 'https://www.amazon.in/s?k={}&ref=nb_sb_noss_2'
    search_term = search_term.replace(' ', '+')
    
    url = generic.format(search_term)
    
    url += '&page{}'
    return url

def extract_record(item):
    atag = item.h2.a
    description = atag.text.strip()
    url = 'https://www.amazon.in/' + atag.get('href')
    
    try:
        price_parent = item.find('span', 'a-price')
        price = price_parent.find('span', 'a-offscreen').text
    
    except AttributeError:
        return
    
    try:
        rating = item.i.text
        reviews = item.find('span', {'class': 'a-size-base', 'dir': 'auto' }).text
        
    except AttributeError:
        rating =''
        reviews = ''
        
    result = (description, price, rating, reviews, url)
    
    return result


def main(search_term):
    options = EdgeOptions()
    options.use_chromium = True
    driver = Edge(options = options)
    
    records = []
    url = get_url(search_term)
    
    for page in range(1, 21):
        driver.get(url.format(page))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        results = soup.find_all('div', {'data-component-type' : 's-search-result'})
        
        for item in results:
            record = extract_record(item)
            if record: 
                records.append(record)
    
    driver.close()
    
    with open('keyboards.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Description', 'Price', 'Rating', 'Reviews', 'Url'])
        writer.writerows(records)
        
        
        
main('keyboard')        
