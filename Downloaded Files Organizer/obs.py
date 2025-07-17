import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent  # Import missing type
from typing import Callable, Optional

def watcher(directory_path: str, callback: Optional[Callable[[str, str, str], None]] = None) -> None:
    """
    Watches a specified directory for file creation events and processes new files.
    
    Args:
        directory_path: The path to the directory to monitor.
        callback: Optional callback function to handle the file processing. 
                Defaults to the internal add_to_dir function.
    """
    # Import add_to_dir here if not available globally
    try:
        from move_to_directory import add_to_dir
    except ImportError:
        def add_to_dir(extension: str, file_path: str, target_dir: str) -> None:
            """Default implementation (replace with actual import if available)"""
            print(f"Processing file: {file_path}, Extension: {extension}")

    class FileCreationHandler(FileSystemEventHandler):
        def on_created(self, event: FileSystemEvent) -> None:
            """Handles file creation events."""
            if not event.is_directory:
                file_path = event.src_path
                file_extension = os.path.splitext(file_path)[1].lstrip('.').lower()
                
                # Allow file to fully write before processing
                time.sleep(2)
                
                # Process the new file
                handler = callback or add_to_dir
                handler(file_extension, file_path, directory_path)

    # Setup and start the file system observer
    event_handler = FileCreationHandler()
    observer = Observer()
    observer.schedule(event_handler, directory_path, recursive=True)
    observer.start()
    
    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()    