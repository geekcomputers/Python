# Author : RIZWAN AHMAD


# pip3 install requests

import requests


def download(url, filename):
    try:
        with requests.get(url, stream=True, timeout=10) as response:
            response.raise_for_status()  # Raises error for 4xx/5xx

            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

        print(f"Successfully downloaded: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")


# Example usage
url = "https://avatars0.githubusercontent.com/u/29729380?s=400&v=4"
download(url, "avatar.jpg")

