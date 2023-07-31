from pytube import YouTube

# Location where you save the videos.
SAVE_PATH = "E:/"

# List of video links.
links = [
    "https://www.youtube.com/watch?v=p8FuTenSWPI",
    "https://www.youtube.com/watch?v=JWbnEt3xuos",
]

def download_videos(links):
    for link in links:
        try:
            yt = YouTube(link)
        except Exception as e:
            print(f"Connection Error: {e}")
            continue

        # Check files with "mp4" extension
        mp4files = yt.filter(file_extension="mp4")
        video_resolution = mp4files[-1].resolution

        try:
            video_stream = yt.streams.get_by_resolution(video_resolution)
            video_stream.download(SAVE_PATH)
        except Exception as e:
            print(f"Download Error: {e}")
            continue

    print("Task Completed!")

download_videos(links)
