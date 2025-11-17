import requests
import xlwt
from xlwt import Workbook

BASE_URL = 'https://remoteok.com/api'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
REQUEST_HEADER = {
    'User-Agent': USER_AGENT,
    'Accept-Language': 'en-US, en;q=0.5',
}

def get_job_postings():
    """Fetch job postings from RemoteOK API."""
    try:
        res = requests.get(BASE_URL, headers=REQUEST_HEADER)
        res.raise_for_status()
        data = res.json()
        return data[1:]
    except requests.RequestException as e:
        print("Error fetching jobs:", e)
        return []

def save_jobs_to_excel(jobs, filename='remoteok_jobs.xls'):
    """Save job postings to an Excel file."""
    if not jobs:
        print("No job data to save.")
        return
    
    wb = Workbook()
    sheet = wb.add_sheet('Jobs')

    headers = list(jobs[0].keys())
    for col, header in enumerate(headers):
        sheet.write(0, col, header)
        
    for row, job in enumerate(jobs, start=1):
        for col, key in enumerate(headers):
            sheet.write(row, col, str(job.get(key, '')))

    wb.save(filename)
    print(f"Jobs saved to {filename}")

if __name__ == '__main__':
    jobs = get_job_postings()
    save_jobs_to_excel(jobs)
