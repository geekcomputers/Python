'''
 Python program that uses the YouTube Data API to fetch the top 10 trending YouTube videos. 
You’ll need to have an API key from Google Cloud Platform to use the YouTube Data API.

First, install the google-api-python-client library if you haven’t already: 
pip install google-api-python-client

Replace 'YOUR_API_KEY' with your actual API key. This script will fetch and print the titles, 
channels, and view counts of the top 10 trending YouTube videos in India. 
You can change the regionCode to any other country code if needed.

Then, you can use the following code:

'''

from googleapiclient.discovery import build

# Replace with your own API key
API_KEY = 'YOUR_API_KEY'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def get_trending_videos():
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    
    # Call the API to get the top 10 trending videos
    request = youtube.videos().list(
        part='snippet,statistics',
        chart='mostPopular',
        regionCode='IN',  # Change this to your region code
        maxResults=10
    )
    response = request.execute()
    
    # Print the video details
    for item in response['items']:
        title = item['snippet']['title']
        channel = item['snippet']['channelTitle']
        views = item['statistics']['viewCount']
        print(f'Title: {title}\nChannel: {channel}\nViews: {views}\n')

if __name__ == '__main__':
    get_trending_videos()
