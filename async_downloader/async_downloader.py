"""
It's example of usage asyncio+aiohttp to downloading.

You should install aiohttp for using:
(You can use virtualenv to testing)

pip install -r /path/to/requirements.txt

"""

import asyncio
import aiohttp
from os.path import basename
import concurrent.futures as cofu


def download(ways):

    if not ways:
        print('Ways list is empty. Downloading is impossible')
        return

    print('downloading..')

    success_files = set()
    failure_files = set()

    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(
            async_downloader(ways, event_loop, success_files, failure_files)
        )
    finally:
        event_loop.close()

    print('Download complete')
    print('-' * 100)

    if success_files:
        print('success:')
        for file in success_files:
            print(file)

    if failure_files:
        print('failure:')
        for file in failure_files:
            print(file)


async def async_downloader(ways, loop, success_files=set(), failure_files=set()):

    async with aiohttp.ClientSession() as session:
        coroutines = [
            download_file_by_url(
                url,
                session=session,
            ) for url in ways]

        completed, pending = await asyncio.wait(coroutines, return_when=cofu.FIRST_COMPLETED)
        while pending:

            for task in completed:
                fail, url = task.result()

                if fail:
                    failure_files.add(url)
                else:
                    success_files.add(url)

            completed, pending = await asyncio.wait(pending, return_when=cofu.FIRST_COMPLETED)

        for task in completed:
            fail, url = task.result()
            if fail:
                failure_files.add(url)
            else:
                success_files.add(url)


async def download_file_by_url(url, session=None):

    fail = True
    file_name = basename(url)

    assert session

    try:
        async with session.get(url) as response:
            if response.status == 404:
                print('\t{} from {} : Failed : {}'.format(file_name, url, '404 - Not found'))
                return fail, url

            if not response.status == 200:
                print('\t{} from {} : Failed : HTTP response {}'.format(file_name, url, response.status))
                return fail, url

            data = await response.read()

            with open(file_name, 'wb') as file:
                file.write(data)

    except asyncio.TimeoutError as err:
        print('\t{} from {}: Failed : {}'.format(file_name, url, 'Timeout error'))

    except aiohttp.client_exceptions.ClientConnectionError as err:
        print('\t{} from {}: Failed : {}'.format(file_name, url, 'Client connection error'))

    else:
        print('\t{} from {} : Success'.format(file_name, url))
        fail = False

    return fail, url


def test():

    ways = ['https://www.wikipedia.org',
            'https://www.ya.ru',
            'https://www.duckduckgo.com',
            'https://www.fail-path.unknown',
            ]

    download(ways)


if __name__ == "__main__":
    test()
