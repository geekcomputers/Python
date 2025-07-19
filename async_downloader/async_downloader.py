"""
Example of using asyncio and aiohttp for asynchronous downloads.

Requires:
- aiohttp library for asynchronous HTTP requests
- Python 3.13.5 or compatible

Install dependencies with:
pip install aiohttp
"""

import asyncio
from os.path import basename

import aiohttp
from aiohttp import ClientError, ClientSession


def download(urls: list[str]) -> None:
    """
    Download files from given URLs asynchronously.
    
    Args:
        urls: List of URLs to download
        
    Prints download statistics and results upon completion.
    """
    if not urls:
        print("URL list is empty. Downloading aborted.")
        return

    print("Initiating downloads...")

    success_files: set[str] = set()
    failure_files: set[str] = set()

    # Run async downloader in the current event loop
    asyncio.run(
        async_downloader(urls, success_files, failure_files)
    )

    print("Download process completed")
    print("-" * 100)

    if success_files:
        print("Successful downloads:")
        for file in success_files:
            print(file)

    if failure_files:
        print("Failed downloads:")
        for file in failure_files:
            print(file)

async def async_downloader(urls: list[str], success_files: set[str], failure_files: set[str]) -> None:
    """
    Asynchronous downloader that processes multiple URLs concurrently.
    
    Args:
        urls: List of URLs to download
        success_files: Set to store successfully downloaded URLs
        failure_files: Set to store failed URLs
        
    Uses aiohttp ClientSession for efficient asynchronous requests.
    """
    async with ClientSession() as session:
        # Create and execute download tasks concurrently
        tasks = [
            download_file_by_url(url, session)
            for url in urls
        ]
        
        # Process results as tasks complete
        for task in asyncio.as_completed(tasks):
            failed, url = await task
            
            if failed:
                failure_files.add(url)
            else:
                success_files.add(url)

async def download_file_by_url(url: str, session: ClientSession) -> tuple[bool, str]:
    """
    Download a single file from given URL using provided session.
    
    Args:
        url: URL of the file to download
        session: aiohttp ClientSession object
        
    Returns:
        Tuple containing:
        - bool: True if download failed, False if successful
        - str: Original URL
        
    Handles various HTTP errors and network exceptions.
    """
    failed = True
    file_name = basename(url)

    try:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise exception for 4xx/5xx status codes
            
            # Read response data
            data = await response.read()
            
            # Write data to file
            with open(file_name, "wb") as file:
                file.write(data)

    except aiohttp.ClientResponseError as e:
        print(f"\t{file_name} from {url}: Failed - HTTP Error {e.status}")
        
    except TimeoutError:
        print(f"\t{file_name} from {url}: Failed - Connection timed out")
        
    except ClientError as e:
        print(f"\t{file_name} from {url}: Failed - Connection error: {str(e)}")
        
    except Exception as e:
        print(f"\t{file_name} from {url}: Failed - Unexpected error: {str(e)}")
        
    else:
        print(f"\t{file_name} from {url}: Success")
        failed = False

    return failed, url

def test() -> None:
    """
    Test the download functionality with sample URLs.
    """
    urls = [
        "https://www.wikipedia.org",
        "https://www.ya.ru",
        "https://www.duckduckgo.com",
        "https://www.fail-path.unknown",  # Intentionally invalid URL
    ]

    download(urls)

if __name__ == "__main__":
    test()