import pytube
import sys
import os
import shutil
import logging
import time
from colorama import init, Fore, Style
from typing import Optional, List
from pytube.exceptions import RegexMatchError, VideoUnavailable
from pytube import Playlist
from math import ceil

init(autoreset=True)


class PlaylistHandler:
    """A class to handle downloading videos from a YouTube playlist."""

    def __init__(self, downloader: "YouTubeDownloader", playlist_url: str):
        """Initializes the playlist handler with the downloader and playlist URL."""
        self.downloader = downloader
        self.playlist_url = playlist_url
        self.playlist = self._get_playlist()

    def _get_playlist(self) -> Playlist:
        """Fetches the YouTube playlist using pytube and logs errors."""
        try:
            playlist = Playlist(self.playlist_url)
            self.downloader.logger.info(f"Playlist fetched: {playlist.title}")
            return playlist
        except RegexMatchError:
            self.downloader._print_error(
                "Invalid playlist URL. Please enter a valid URL."
            )
            raise
        except Exception as e:
            self.downloader.logger.error(f"Error fetching playlist: {e}")
            self.downloader._print_error(f"Error fetching playlist: {e}")
            raise

    def _get_videos_to_download(self) -> List[pytube.YouTube]:
        """Allows the user to select videos to download, or to download all videos in playlist."""
        videos = self.playlist.videos
        num_videos = len(videos)

        while True:
            self.downloader._print_menu_header(f"Select videos from playlist: {self.playlist.title}")
            print(f"Total videos: {num_videos}")
            print("1. Download all videos.")
            print("2. Select specific videos.")
            print("3. Go Back to main menu.")

            choice = input(f"{Fore.YELLOW}Choose an option: {Style.RESET_ALL}")

            if choice == "1":
                self.downloader.logger.info("Selected to download all videos in playlist.")
                return videos
            elif choice == "2":
                return self._select_specific_videos(videos)
            elif choice == "3":
                self.downloader.main_menu()
                return []  # Returns empty list to signify "go back".
            else:
                self.downloader._print_error("Invalid option, try again.")


    def _select_specific_videos(self, videos: List[pytube.YouTube]) -> List[pytube.YouTube]:
        """Handles the selection of specific videos for download."""
        num_videos = len(videos)
        while True:
            self.downloader._print_menu_header("Select Specific Videos (Enter comma separated indexes or 'b' to go back)")
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video.title}")

            indexes = input(f"{Fore.YELLOW}Enter video numbers separated by commas: {Style.RESET_ALL}")

            if indexes.lower() == 'b':
                self.downloader.main_menu()
                return []

            try:
                selected_indexes = [int(index.strip()) -1 for index in indexes.split(',')]
                if all(0 <= i < num_videos for i in selected_indexes):
                    self.downloader.logger.info(f"Selected specific videos for download : {selected_indexes}")
                    return [videos[i] for i in selected_indexes]
                else:
                    self.downloader._print_error("Invalid video numbers, please try again")
            except ValueError:
                self.downloader._print_error("Invalid input, please enter valid video numbers separated by comma.")
            except Exception as e:
                self.downloader._print_error(f"An unexpected error occurred : {e}")

    def download_playlist(self):
        """Downloads the videos in the playlist."""
        videos = self._get_videos_to_download()
        if videos:
            for i, video in enumerate(videos, 1):
                self.downloader.logger.info(f"Downloading video : {video.title}")
                self.downloader._print_menu_header(f"Downloading : {video.title} ({i} of {len(videos)})")
                try:
                    stream = self.downloader._select_stream("progressive", video)
                    if stream:
                        if self.downloader._confirm_download(stream, "video"):
                          self.downloader._start_download(stream, "video")
                except Exception as e:
                    self.downloader.logger.error(f"Error downloading video {video.title} : {e}")
                    self.downloader._print_error(f"Error downloading video {video.title} : {e}")
            self.downloader._print_success("✅ All videos downloaded from playlist!")
        return None


class YouTubeDownloader:
    """A class for downloading YouTube videos and audios with logging and error handling."""

    def __init__(self):
        """Initializes the downloader and logger with url input."""
        self.logger = self._setup_logger()
        self.url = self._get_url()
        try:
            self.youtube = pytube.YouTube(
                self.url, on_progress_callback=self._on_progress
            )
        except RegexMatchError:
             self.logger.error(f"Invalid URL provided : {self.url}")
             self._print_error("\n❌ Invalid URL. Please enter a valid URL.\n")
             self.__init__()  # Restart for a new URL if it's invalid
             return
        except VideoUnavailable:
            self.logger.error(f"Video not available : {self.url}")
            self._print_error(f"\n❌ Video unavailable for the URL : {self.url}\n")
            self.__init__()
            return
        except Exception as e:
             self.logger.error(f"An unexpected error occurred : {e}")
             self._print_error(f"\n❌ An error occurred: {e}\n")
             sys.exit(1)

        self.file_size = 0
        self.default_path = os.path.join(os.path.expanduser("~"), "Downloads")
        self.main_menu()

    def _setup_logger(self) -> logging.Logger:
        """Sets up the logger for the app."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG) # Log all level of log message

        # Log file
        file_handler = logging.FileHandler("youtube_downloader.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(levelname)s - %(message)s"
        )  # Keep console logs cleaner
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


    def _get_terminal_width(self) -> int:
        """Gets terminal width, defaults to 80 if not determinable."""
        try:
            return shutil.get_terminal_size().columns
        except (AttributeError, OSError):
            return 80

    def _center_text(self, text: str, fill_char: str = " ") -> str:
        """Centers text based on terminal width."""
        width = self._get_terminal_width()
        padding = (width - len(text)) // 2
        return fill_char * padding + text + fill_char * padding

    def _print_menu_header(self, title: str) -> None:
        """Prints a formatted menu header."""
        width = self._get_terminal_width()
        border = "-" * width
        print(f"\n{Fore.CYAN}{border}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{self._center_text(title)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{border}{Style.RESET_ALL}")

    def _print_error(self, message: str) -> None:
        """Prints an error message."""
        print(f"{Fore.RED}{message}{Style.RESET_ALL}")

    def _print_success(self, message: str) -> None:
        """Prints a success message."""
        print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

    def _get_url(self) -> str:
        """Gets the video URL from the user."""
        while True:
            url = input(f"{Fore.YELLOW}Enter the video or playlist URL: {Style.RESET_ALL}")
            if url:
                return url
            else:
                self._print_error("Please enter the URL.")

    def _select_stream(
        self, stream_type: str, video: Optional[pytube.YouTube] = None
    ) -> Optional[pytube.Stream]:
        """Displays stream options and allows the user to select a stream."""
        target = video or self.youtube # Use video if available or youtube if not
        self._print_menu_header(f"Available {stream_type.title()} Streams")
        streams = target.streams.filter(type=stream_type)
        if not streams:
            self._print_error(f"❌ No {stream_type} streams found for this video.")
            return None
        print("Available streams:")
        for i, stream in enumerate(streams, 1):
            print(
                f"{i}. Resolution: {stream.resolution or 'N/A'}, FPS: {stream.fps or 'N/A'}, Type: {stream.mime_type}"
            )

        while True:
            try:
                choice = input(f"Choose a {stream_type} stream number ('b' to back): ")
                if choice.lower() == "b":
                    self.main_menu()
                    return None
                choice = int(choice)
                if 1 <= choice <= len(streams):
                    return streams[choice - 1]
                else:
                    self._print_error("Invalid stream number. Try again.")
            except ValueError:
                self._print_error("Invalid input. Enter a number or 'b'.")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during select stream : {e}")
                self._print_error(f"An unexpected error occurred during select stream : {e}")
                return None

    def _confirm_download(self, stream: pytube.Stream, stream_type: str) -> bool:
        """Confirms the download with the user."""
        self.file_size = stream.filesize / 1000000
        self._print_menu_header(f"{stream_type.title()} Download Confirmation")
        print(
            f"""
            Title   : {Fore.GREEN}{self.youtube.title}{Style.RESET_ALL}
            Author  : {Fore.GREEN}{self.youtube.author}{Style.RESET_ALL}
            Size    : {Fore.GREEN}{self.file_size:.2f} MB{Style.RESET_ALL}
            Resolution: {Fore.GREEN}{stream.resolution if stream.resolution else 'N/A'}{Style.RESET_ALL}
            FPS     : {Fore.GREEN}{stream.fps if stream.fps else 'N/A'}{Style.RESET_ALL}
            Location: {Fore.YELLOW}{self.default_path}{Style.RESET_ALL}
            """
        )
        while True:
            confirmation = input("Confirm download (y/n)?: ").lower()
            if confirmation == "y":
                return True
            elif confirmation == "n":
                self.main_menu()
                return False
            else:
                self._print_error("Invalid input, enter 'y' or 'n'.")

    def _start_download(self, stream: pytube.Stream, stream_type: str) -> None:
        """Starts the download process with resume functionality."""
        filename = os.path.join(self.default_path, stream.default_filename)
        temp_filename = filename + ".part"
        try:
            self._print_menu_header(f"Starting {stream_type} Download")
            print("Downloading...")
            # Check for an existing part file
            if os.path.exists(temp_filename):
               file_size = os.path.getsize(temp_filename) # Get the file size if it is already downloaded
               self.logger.info(f"Resuming download from:{temp_filename}")
               stream.download(output_path=self.default_path, filename=temp_filename,  filesize=file_size)
               os.rename(temp_filename, filename) # change the file extension from .part to normal after download
            else:
                self.logger.info(f"Starting new download : {filename}")
                stream.download(output_path=self.default_path, filename=temp_filename)
                os.rename(temp_filename, filename)

            self._print_success(f"\n✅ {stream_type.title()} downloaded to: {Fore.YELLOW}{self.default_path}{Style.RESET_ALL}")
        except KeyboardInterrupt:
            self.logger.error("Download was canceled by user")
            self._print_error("\n❌ Download canceled by user.")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Error during download: {e}")
            self._print_error(f"\n❌ Error during download: {e}\n")


    def _on_progress(self, stream, chunk, bytes_remaining):
       total_size = stream.filesize
       bytes_downloaded = total_size - bytes_remaining
       percent = (bytes_downloaded / total_size) * 100
       width = self._get_terminal_width()
       progress_bar_length = int(width * 0.3)
       progress_filled = int((bytes_downloaded / total_size) * progress_bar_length)
       progress_bar = f"[{'=' * progress_filled}{' ' * (progress_bar_length - progress_filled)}]"

       # speed calculation
       start_time = getattr(self, '_start_time', None) or time.time()
       if start_time: # Avoid division by 0
           elapsed_time = time.time() - start_time
           download_speed = bytes_downloaded / elapsed_time if elapsed_time > 0 else 0
           remaining_time = (bytes_remaining / download_speed) if download_speed > 0 else 0
           setattr(self, '_start_time', time.time())
       else:
           download_speed = 0
           remaining_time = 0

       download_speed_mbps = download_speed / (1024 * 1024)
       remaining_time_min = ceil(remaining_time/60)

       progress_text = f"Downloading... {percent:.2f}% {progress_bar} [{bytes_downloaded / (1024*1024):.2f}MB of {total_size / (1024*1024):.2f}MB] | {download_speed_mbps:.2f} MB/s | ETA: {remaining_time_min}min "
       print(progress_text, end='\r')

    def main_menu(self) -> None:
        """Displays the main menu and handles user interactions."""
        while True:
            self._print_menu_header("Main Menu")
            print("1. Download Video")
            print("2. Download Audio")
            print("3. Download Playlist")
            print("4. Change download path")
            print("5. Quit")

            choice = input(f"{Fore.YELLOW}Choose an option: {Style.RESET_ALL}")
            if choice == "1":
                stream = self._select_stream("progressive")
                if stream and self._confirm_download(stream, "video"):
                    self._start_download(stream, "video")
            elif choice == "2":
                stream = self._select_stream("audio")
                if stream and self._confirm_download(stream, "audio"):
                    self._start_download(stream, "audio")
            elif choice == "3":
                try:
                    playlist_handler = PlaylistHandler(self, self.url)
                    playlist_handler.download_playlist()
                except Exception:
                    pass
            elif choice == "4":
                self._change_path()
            elif choice == "5":
                print("Exiting application.")
                sys.exit(0)
            else:
                self._print_error("Invalid choice. Please try again.")

    def _change_path(self) -> None:
        """Allows the user to change the download path."""
        while True:
            new_path = input(
                f"Current path: {Fore.YELLOW}{self.default_path}{Style.RESET_ALL} | New path (or Enter to keep current): "
            )
            if new_path:
                if os.path.isdir(new_path):
                    self.default_path = new_path
                    print(f"Path updated: {Fore.GREEN}{self.default_path}{Style.RESET_ALL}")
                    return
                else:
                    self._print_error("Invalid path, enter a valid path.")
            else:
                print("Download path stays the same.")
                return


#  Unit test implementation (conceptual)
#  add a separate file for unit testing
#
# def test_download_video():
#     """Test that a video can be downloaded."""
#     downloader = YouTubeDownloader()
#     # set URL to testing video
#     stream = downloader._select_stream("progressive")
#     assert stream is not None
#     # call download function, this will create a test video
#     # assert that the test video exists after download function
# def test_download_audio():
#    """Test that audio only stream is being downloaded"""
#    # implement code to test audio
#
# def test_resume_download():
#    """Test if the resume function works correctly"""
#   # imlement code to test the resume download
#
# # add more tests


#  Documentation :
#  add a README.md with instructions on how to use the application
#  add an API documentation to describe the different methods used by the application


if __name__ == "__main__":
    try:
        YouTubeDownloader()
    except KeyboardInterrupt:
        print("Program terminated by user")
        sys.exit(0)
    except Exception as e:
         print(f"\n❌ An unexpected error occurred: {e}\n")
         sys.exit(1)
