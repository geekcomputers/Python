from bs4 import BeautifulSoup
import requests
import time

# Script Name		: gstin_scraper.py
# Author				: Purshotam
# Created				: Sep 6, 2021 7:59 PM
# Last Modified		: Oct 3, 2023 6:28 PM
# Version				: 1.0
# Modifications		:
""" Description	:
GSTIN, short for Goods and Services Tax Identification Number, 
is a unique 15 digit identification number assigned to every taxpayer 
(primarily dealer or supplier or any business entity) registered under the GST regime.
This script is able to fetch GSTIN numbers for any company registered in the
Mumbai / Banglore region.
"""


# Using a demo list in case of testing the script. 
# This list will be used in case user skips "company input" dialogue by pressing enter.
demo_companies = ["Bank of Baroda", "Trident Limited", "Reliance Limited", "The Yummy Treat", "Yes Bank", "Mumbai Mineral Trading Corporation"]

def get_company_list():
    company_list = []
    
    while True:
        company = input("Enter a company name (or press Enter to finish): ")
        if not company:
            break
        company_list.append(company)
    
    return company_list

def fetch_gstins(company_name, csrf_token):
    third_party_gstin_site = "https://www.knowyourgst.com/gst-number-search/by-name-pan/"
    payload = {'gstnum': company_name, 'csrfmiddlewaretoken': csrf_token}

    # Getting the HTML content and extracting the GSTIN content using BeautifulSoup.
    html_content = requests.post(third_party_gstin_site, data=payload)
    soup = BeautifulSoup(html_content.text, 'html.parser')
    site_results = soup.find_all(id="searchresult")

    # Extracting GSTIN specific values from child elements.
    gstins = [result.strong.next_sibling.next_sibling.string for result in site_results]

    return gstins

def main():
    temp = get_company_list()
    companies = temp if temp else demo_companies

    all_gstin_data = ""
    third_party_gstin_site = "https://www.knowyourgst.com/gst-number-search/by-name-pan/"

    # Getting the CSRF value for further RESTful calls.
    page_with_csrf = requests.get(third_party_gstin_site)
    soup = BeautifulSoup(page_with_csrf.text, 'html.parser')
    csrf_token = soup.find('input', {"name": "csrfmiddlewaretoken"})['value']

    for company in companies:
        gstins = fetch_gstins(company, csrf_token)

        # Only include GSTINs for Bengaluru and Mumbai-based companies
        comma_separated_gstins = ', '.join([g for g in gstins if g.startswith(('27', '29'))])

        all_gstin_data += f"{company} = {comma_separated_gstins}\n\n"

        # Delaying for false DDOS alerts on the third-party site
        time.sleep(0.5)

    # Printing the data
    print(all_gstin_data)

if __name__ == "__main__":
    main()
