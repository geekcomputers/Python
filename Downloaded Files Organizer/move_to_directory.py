import os
import platform
import shutil

# File extension categories and their associated extensions
FILE_EXTENSIONS: dict[str, list[str]] = {
    "web": ["css", "less", "scss", "wasm"],
    "audio": ["aac", "aiff", "ape", "au", "flac", "gsm", "it", "m3u", "m4a", 
             "mid", "mod", "mp3", "mpa", "pls", "ra", "s3m", "sid", "wav", "wma", "xm"],
    "code": ["c", "cc", "class", "clj", "cpp", "cs", "cxx", "el", "go", "h", "java", 
            "lua", "m", "m4", "php", "pl", "po", "py", "rb", "rs", "swift", "vb", 
            "vcxproj", "xcodeproj", "xml", "diff", "patch", "html", "js"],
    "slide": ["ppt", "odp"],
    "sheet": ["ods", "xls", "xlsx", "csv", "ics", "vcf"],
    "image": ["3dm", "3ds", "max", "bmp", "dds", "gif", "jpg", "jpeg", "png", "psd", 
             "xcf", "tga", "thm", "tif", "tiff", "ai", "eps", "ps", "svg", "dwg", 
             "dxf", "gpx", "kml", "kmz", "webp"],
    "archive": ["7z", "a", "apk", "ar", "bz2", "cab", "cpio", "deb", "dmg", "egg", 
               "gz", "iso", "jar", "lha", "mar", "pea", "rar", "rpm", "s7z", "shar", 
               "tar", "tbz2", "tgz", "tlz", "war", "whl", "xpi", "zip", "zipx", "xz", "pak"],
    "book": ["mobi", "epub", "azw1", "azw3", "azw4", "azw6", "azw", "cbr", "cbz"],
    "text": ["doc", "docx", "ebook", "log", "md", "msg", "odt", "org", "pages", "pdf", 
            "rtf", "rst", "tex", "txt", "wpd", "wps"],
    "executable": ["exe", "msi", "bin", "command", "sh", "bat", "crx"],
    "font": ["eot", "otf", "ttf", "woff", "woff2"],
    "video": ["3g2", "3gp", "aaf", "asf", "avchd", "avi", "drc", "flv", "m2v", "m4p", 
             "m4v", "mkv", "mng", "mov", "mp2", "mp4", "mpe", "mpeg", "mpg", "mpv", 
             "mxf", "nsv", "ogg", "ogv", "ogm", "qt", "rm", "rmvb", "roq", "srt", 
             "svi", "vob", "webm", "wmv", "yuv"]
}


def get_unique_filename(base_name: str, extension: str, target_dir: str) -> str:
    """
    Generate a unique filename to avoid overwriting existing files.
    
    Args:
        base_name: Original filename without extension
        extension: File extension (without dot)
        target_dir: Directory where the file will be moved
        
    Returns:
        Unique filename in format "base_name[counter].extension"
    """
    counter = 0
    # Check existing files in target directory
    for filename in os.listdir(target_dir):
        if filename.startswith(base_name) and filename.endswith(f".{extension}"):
            counter += 1
    
    return f"{base_name}{counter}.{extension}" if counter > 0 else f"{base_name}.{extension}"


def move_to_category_dir(extension: str, source_path: str, base_dir: str) -> None:
    """
    Move a file to its corresponding category directory based on extension.
    
    Args:
        extension: File extension to determine category
        source_path: Full path to the source file
        base_dir: Base directory where category folders will be created
    """
    # Get original filename components
    filename = os.path.basename(source_path)
    base_name = filename[:filename.rfind(f".{extension}")]  # Name without extension
    
    # Find matching category for the extension
    target_category = None
    for category, extensions in FILE_EXTENSIONS.items():
        if extension in extensions:
            target_category = category
            break
    
    if not target_category:
        return  # Skip uncategorized files
    
    # Create target directory path
    target_dir = os.path.join(base_dir, target_category)
    
    try:
        # Create category directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Move file to target directory
        destination_path = os.path.join(target_dir, filename)
        
        # Handle existing files by renaming
        if os.path.exists(destination_path):
            new_filename = get_unique_filename(base_name, extension, target_dir)
            new_destination = os.path.join(target_dir, new_filename)
            shutil.move(source_path, new_destination)
        else:
            shutil.move(source_path, destination_path)
    
    except shutil.Error as e:
        print(f"Error moving file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Example usage:
if __name__ == "__main__":
    # Detect the operating system
    current_os = platform.system()
    
    # Set base directory based on the detected OS
    if current_os == 'Windows' or current_os in ['Linux', 'Darwin']:
        base_directory = os.path.join(os.path.expanduser('~'), 'Downloads')
    else:
        print(f"Unsupported operating system: {current_os}")
        base_directory = os.getcwd()  # Use current directory as fallback
    
    print(f"Sorting files in: {base_directory}")
    
    # Process all files in the base directory
    try:
        for item in os.listdir(base_directory):
            item_path = os.path.join(base_directory, item)
            
            # Skip directories, process only files
            if os.path.isfile(item_path):
                # Get file extension (lowercase)
                if "." in item:
                    ext = item.split(".")[-1].lower()
                    move_to_category_dir(ext, item_path, base_directory)
    except Exception as e:
        print(f"Error processing directory: {e}")    