"""
It's example of usage asyncio+aiohttp to downloading.
You should install aiohttp for using:
(You can use virtualenv to testing)
pip install -r /path/to/requirements.txt
"""

import asyncio
from os.path import basename

import aiohttp


def download(ways):
    """
    Download all files from the given list of URLs.

    Args:
        ways (list): A list of URL strings to download.

    Prints progress and final summary of succeeded/failed downloads.
    """
    if not ways:
        print("Ways list is empty. Downloading is impossible")
        return

    print("downloading..")

    success_files = set()
    failure_files = set()

    # asyncio.run() creates a new event loop, runs the coroutine,
    # and closes the loop automatically – this fixes the
    # "no current event loop" error in Python 3.10+.
    asyncio.run(async_downloader(ways, success_files, failure_files))

    print("Download complete")
    print("-" * 100)

    if success_files:
        print("success:")
        for file in success_files:
            print(file)

    if failure_files:
        print("failure:")
        for file in failure_files:
            print(file)


async def async_downloader(ways, success_files, failure_files):
    """
    Asynchronously download multiple files using aiohttp.

    Args:
        ways (list): List of URL strings.
        success_files (set): Set to collect successful URLs.
        failure_files (set): Set to collect failed URLs.
    """
    async with aiohttp.ClientSession() as session:
        # Create a coroutine for each URL
        coroutines = [download_file_by_url(url, session=session) for url in ways]

        # Process tasks as they complete
        for task in asyncio.as_completed(coroutines):
            fail, url = await task
            if fail:
                failure_files.add(url)
            else:
                success_files.add(url)


async def download_file_by_url(url, session=None):
    """
    Download a single file from a URL and save it locally.

    Args:
        url (str): The URL to download.
        session (aiohttp.ClientSession): The session to use for the request.

    Returns:
        tuple: (fail, url) where fail is True if download failed, else False.
    """
    fail = True
    file_name = basename(url)

    # Ensure a valid session is provided
    assert session, "aiohttp session is required"

    try:
        async with session.get(url) as response:
            # Handle 404 specifically
            if response.status == 404:
                print(f"\t{file_name} from {url} : Failed : 404 - Not found")
                return fail, url

            # Any non-200 status is considered a failure
            if response.status != 200:
                print(
                    f"\t{file_name} from {url} : Failed : HTTP response {response.status}"
                )
                return fail, url

            # Read and save the content
            data = await response.read()
            with open(file_name, "wb") as file:
                file.write(data)

    except asyncio.TimeoutError:
        print(f"\t{file_name} from {url}: Failed : Timeout error")

    except aiohttp.client_exceptions.ClientConnectionError:
        print(f"\t{file_name} from {url}: Failed : Client connection error")

    else:
        # No exception occurred – download succeeded
        print(f"\t{file_name} from {url} : Success")
        fail = False

    return fail, url


def test():
    """Test the downloader with a list of sample URLs."""
    ways = [
        "https://www.wikipedia.org",
        "https://www.ya.ru",
        "https://www.duckduckgo.com",
        "https://www.fail-path.unknown",
    ]
    download(ways)


if __name__ == "__main__":
    test()
