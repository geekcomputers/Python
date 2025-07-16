# Instagram Monitoring Tool with Enhanced Error Handling and Default Parameters
import asyncio
import multiprocessing
import os
import random
import socket
import sys
import time
import httpx
import traceback

import bs4
import pymongo

import socks
import ujson

try:
    import con_file as config
except ImportError:
    import insta_monitering.con_file as config


class ProxyApplyingDecorator:
    """
    Decorator class for applying SOCKS5 proxy to HTTP requests.
    
    Attributes:
        proxy_file (str): Path to the file containing proxy list.
        _IP (str): Selected proxy IP address and port.
    """
    def __init__(self, proxy_file="ipList.txt"):
        """
        Initialize the proxy decorator.
        
        Args:
            proxy_file (str, optional): Path to the proxy list file. Defaults to "ipList.txt".
        """
        self.proxy_file = proxy_file
        self._IP = self._get_random_proxy()

    def _get_random_proxy(self):
        """
        Read proxy list from file and select a random proxy.
        
        Returns:
            str: Randomly selected proxy in "host:port" format.
        """
        try:
            if not os.path.exists(self.proxy_file):
                raise FileNotFoundError(f"Proxy file not found: {self.proxy_file}")
                
            with open(self.proxy_file, "r") as f:
                ipdata = f.read().strip()
                
            if not ipdata:
                raise ValueError("Proxy file is empty")
                
            return random.choice(ipdata.split(","))
        except Exception as e:
            print(f"Error loading proxy: {e}")
            return None

    def __call__(self, function_to_apply_proxy):
        """
        Wrap the target function with proxy configuration.
        
        Args:
            function_to_apply_proxy (callable): Function to be wrapped.
            
        Returns:
            callable: Wrapped function with proxy applied.
        """
        def wrapper_function(url):
            if not self._IP:
                print("No proxy available, using direct connection")
                return function_to_apply_proxy(url)
                
            original_socket = socket.socket
            try:
                proxy_parts = self._IP.split(":")
                proxy_host = proxy_parts[0]
                proxy_port = int(proxy_parts[1]) if len(proxy_parts) > 1 else config.SOCKS5_PROXY_PORT
                
                # Configure SOCKS5 proxy
                socks.set_default_proxy(
                    socks.SOCKS5,
                    proxy_host,
                    proxy_port,
                    True,
                    config.auth,
                    config.passcode,
                )
                socket.socket = socks.socksocket
                return function_to_apply_proxy(url)
            except Exception as e:
                print(f"Proxy error: {e}")
                return function_to_apply_proxy(url)  # Fallback to direct connection
            finally:
                # Restore original socket configuration
                socket.socket = original_socket
                socks.set_default_proxy()

        return wrapper_function


async def process_post_data(html_data):
    """
    Process HTML data of a single Instagram post to extract metadata.
    
    Args:
        html_data (str): HTML content of the post page.
        
    Returns:
        dict: Extracted post metadata including ID, owner, location, etc.
    """
    if not html_data:
        return {}
        
    try:
        soup = bs4.BeautifulSoup(html_data, "html.parser")
        scripts = soup.find_all("script", {"type": "text/javascript"})
        
        # Find the script containing post data
        for script in scripts:
            if "window._sharedData =" in script.text:
                json_text = script.text.strip()
                json_text = json_text.replace("window._sharedData =", "").replace(";", "").strip()
                data = ujson.loads(json_text)
                
                # Extract post details
                if "entry_data" in data and "PostPage" in data["entry_data"]:
                    post = data["entry_data"]["PostPage"][0]["graphql"]["shortcode_media"]
                    return {
                        "id": post.get("id"),
                        "owner": post.get("owner"),
                        "location": post.get("location"),
                        "caption": post.get("edge_media_to_caption", {}).get("edges", [{}])[0].get("node", {}).get("text"),
                        "timestamp": post.get("taken_at_timestamp"),
                    }
                    
        return {}
    except Exception as e:
        print(f"Error processing post data: {e}")
        traceback.print_exc()
        return {}


async def fetch_post_details(url, max_retries=3):
    """
    Asynchronously fetch and process details of an Instagram post.
    
    Args:
        url (str): URL of the Instagram post.
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        
    Returns:
        dict: Processed post data.
    """
    retries = 0
    while retries < max_retries:
        @ProxyApplyingDecorator()
        async def fetch_with_proxy(url):
            """Fetch URL content with proxy applied"""
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                }
                async with httpx.AsyncClient(verify=False, timeout=30) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    return response.text
            except Exception as e:
                print(f"Request failed: {e}")
                return None
                
        html_data = await fetch_with_proxy(url)
        if html_data:
            return await process_post_data(html_data)
            
        retries += 1
        print(f"Retry {retries}/{max_retries} for {url}")
        await asyncio.sleep(2)
        
    return {}


class MonitoringClass:
    """
    Main class for monitoring Instagram hashtags or profiles.
    
    Attributes:
        user (str): User identifier for the monitoring task.
        tags (str): Hashtag or profile name to monitor.
        monitor_type (str): Type of monitoring ("hashtags" or "profile").
        product_id (str): Product identifier for database naming.
        url (str): Instagram URL to fetch data from.
        client (pymongo.MongoClient): MongoDB client.
        db (pymongo.Database): MongoDB database.
        collection (pymongo.Collection): MongoDB collection for storing posts.
    """
    def __init__(
        self, 
        user, 
        tags, 
        monitor_type="hashtags", 
        product_id="insta_monitor",
        db_host=None,
        db_port=None
    ):
        """
        Initialize the monitoring class.
        
        Args:
            user (str): User identifier.
            tags (str): Hashtag or profile name.
            monitor_type (str, optional): Monitoring type. Defaults to "hashtags".
            product_id (str, optional): Product identifier. Defaults to "insta_monitor".
            db_host (str, optional): MongoDB host. Defaults to config.host.
            db_port (int, optional): MongoDB port. Defaults to config.mongoPort.
        """
        self.user = user
        self.tags = tags
        self.monitor_type = monitor_type
        self.product_id = product_id
        
        # Use config values or defaults
        self.db_host = db_host or config.host
        self.db_port = db_port or config.mongoPort
        
        self.client = None
        self.db = None
        self.collection = None
        
        self._initialize()

    def _initialize(self):
        """Initialize MongoDB connection and monitoring URL"""
        try:
            self.client = pymongo.MongoClient(host=self.db_host, port=self.db_port, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')  # Test connection
            
            db_name = f"{self.product_id}:{self.user}:insta"
            self.db = self.client[db_name]
            self.collection = self.db[self.tags]
            
            # Create unique index to prevent duplicate posts
            self.collection.create_index("id", unique=True)
            
            # Build URL based on monitoring type
            if self.monitor_type == "hashtags":
                self.url = f"https://www.instagram.com/explore/tags/{self.tags}/?__a=1"
            elif self.monitor_type == "profile":
                self.url = f"https://www.instagram.com/{self.tags}/?__a=1"
            else:
                raise ValueError(f"Invalid monitor type: {self.monitor_type}. Must be 'hashtags' or 'profile'.")
                
            print(f"Monitoring initialized for {self.monitor_type}: {self.tags}")
        except Exception as e:
            print(f"Initialization error: {e}")
            traceback.print_exc()
            self._cleanup()
            raise

    def _cleanup(self):
        """Clean up resources (close MongoDB connection)"""
        if self.client:
            self.client.close()

    def _parse_instagram_response(self, response_data):
        """
        Parse Instagram API response to extract post information.
        
        Args:
            response_data (dict): JSON response from Instagram API.
            
        Returns:
            list: List of posts with basic information.
        """
        try:
            if not isinstance(response_data, dict):
                raise TypeError("Invalid response format")
                
            if self.monitor_type == "hashtags":
                if "graphql" not in response_data or "hashtag" not in response_data["graphql"]:
                    raise ValueError("Invalid hashtag response structure")
                    
                hashtag_data = response_data["graphql"]["hashtag"]
                posts = []
                
                # Extract recent posts
                edge_media = hashtag_data.get("edge_hashtag_to_media", {})
                for edge in edge_media.get("edges", []):
                    node = edge.get("node", {})
                    posts.append({
                        "id": node.get("id"),
                        "shortcode": node.get("shortcode"),
                        "timestamp": node.get("taken_at_timestamp"),
                        "owner_id": node.get("owner", {}).get("id"),
                        "caption": node.get("edge_media_to_caption", {}).get("edges", [{}])[0].get("node", {}).get("text"),
                        "url": f"https://www.instagram.com/p/{node.get('shortcode')}/"
                    })
                    
                # Extract top posts
                edge_top_posts = hashtag_data.get("edge_hashtag_to_top_posts", {})
                for edge in edge_top_posts.get("edges", []):
                    node = edge.get("node", {})
                    posts.append({
                        "id": node.get("id"),
                        "shortcode": node.get("shortcode"),
                        "timestamp": node.get("taken_at_timestamp"),
                        "owner_id": node.get("owner", {}).get("id"),
                        "caption": node.get("edge_media_to_caption", {}).get("edges", [{}])[0].get("node", {}).get("text"),
                        "url": f"https://www.instagram.com/p/{node.get('shortcode')}/"
                    })
                    
                return posts
                
            elif self.monitor_type == "profile":
                if "graphql" not in response_data or "user" not in response_data["graphql"]:
                    raise ValueError("Invalid profile response structure")
                    
                user_data = response_data["graphql"]["user"]
                posts = []
                
                # Extract user posts
                edge_media = user_data.get("edge_owner_to_timeline_media", {})
                for edge in edge_media.get("edges", []):
                    node = edge.get("node", {})
                    posts.append({
                        "id": node.get("id"),
                        "shortcode": node.get("shortcode"),
                        "timestamp": node.get("taken_at_timestamp"),
                        "owner_id": node.get("owner", {}).get("id"),
                        "caption": node.get("edge_media_to_caption", {}).get("edges", [{}])[0].get("node", {}).get("text"),
                        "url": f"https://www.instagram.com/p/{node.get('shortcode')}/"
                    })
                    
                return posts
                
            return []
        except Exception as e:
            print(f"Error parsing Instagram response: {e}")
            traceback.print_exc()
            return []

    async def _fetch_and_process_posts(self, posts):
        """
        Asynchronously fetch and process detailed information for all posts.
        
        Args:
            posts (list): List of posts with basic information.
            
        Returns:
            list: List of posts with detailed information.
        """
        if not posts:
            return []
            
        print(f"Fetching details for {len(posts)} posts")
        
        # Create event loop
        loop = asyncio.get_event_loop()
        
        # Create tasks for each post
        tasks = []
        for post in posts:
            tasks.append(fetch_post_details(post["url"]))
            
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Merge results into original posts
        for i, result in enumerate(results):
            posts[i].update(result)
            
        return posts

    def _save_to_database(self, posts):
        """
        Save posts to MongoDB database.
        
        Args:
            posts (list): List of posts to save.
        """
        if not posts:
            print("No posts to save")
            return
            
        saved_count = 0
        for post in posts:
            if not post.get("id"):
                continue
                
            try:
                # Add insertion timestamp
                post["_inserted_at"] = time.time()
                
                # Use update_one with upsert to avoid duplicates
                result = self.collection.update_one(
                    {"id": post["id"]},
                    {"$set": post},
                    upsert=True
                )
                
                if result.upserted_id or result.modified_count > 0:
                    saved_count += 1
                    
            except Exception as e:
                print(f"Error saving post {post.get('id')}: {e}")
                
        print(f"Saved {saved_count}/{len(posts)} posts to database")
def run_monitoring(self, max_retries=3):
    """
    Execute the Instagram monitoring task with retry mechanism.
    
    Args:
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        
    Returns:
        bool: True if monitoring completed successfully, False otherwise.
    """
    # Retry loop with improved readability and error handling
    for attempt in range(max_retries):
        try:
            # Fetch Instagram API data using proxy decorator
            response_data = self._fetch_instagram_data()
            if not response_data:
                raise ValueError("Failed to fetch data from Instagram API")
                
            # Parse the response to extract post information
            posts = self._parse_instagram_response(response_data)
            if not posts:
                print("No posts found to process")
                return False
                
            print(f"Processing {len(posts)} posts from {self.tags}")
            
            # Asynchronously process posts and save to database
            self._process_and_save_posts(posts)
            
            print(f"Monitoring completed successfully for {self.tags}")
            return True
            
        except Exception as e:
            # Log detailed error information for debugging
            print(f"Monitoring attempt {attempt + 1}/{max_retries} failed: {e}")
            traceback.print_exc()
            
            # Wait before retrying, except on final attempt
            if attempt < max_retries - 1:
                print(f"Retrying in {config.RETRY_DELAY} seconds...")
                time.sleep(config.RETRY_DELAY)
                
    print(f"Max retries reached ({max_retries}). Monitoring failed.")
    return False

def _fetch_instagram_data(self):
    """
    Fetch data from Instagram API with proxy applied.
    
    Returns:
        dict: JSON response from Instagram, or None on error.
    """
    @ProxyApplyingDecorator()
    def fetch_with_proxy(url):
        """Inner function to apply proxy and handle HTTP request"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        try:
            with httpx.Client(verify=False, timeout=30) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Network request error: {e}")
            return None
            
    return fetch_with_proxy(self.url)

async def _fetch_and_process_post_details(self, posts):
    """
    Asynchronously fetch detailed information for multiple Instagram posts.
    
    Args:
        posts (list): List of posts with basic information (id, shortcode, url)
        
    Returns:
        list: List of dictionaries with detailed post information
    """
    # Create async tasks for each post URL
    tasks = [fetch_post_details(post["url"]) for post in posts]
    
    # Execute all tasks concurrently and return results
    return await asyncio.gather(*tasks)

def _process_and_save_posts(self, posts):
    """
    Process posts asynchronously and save results to MongoDB.
    
    Args:
        posts (list): List of posts to process and save
    """
    # Use asyncio.run() to handle the event loop lifecycle
    # This is preferred over manually managing the event loop in Python 3.7+
    results = asyncio.run(self._fetch_and_process_post_details(posts))
    
    # Merge detailed results back into original post objects
    for i, result in enumerate(results):
        posts[i].update(result)
    
    # Save all processed posts to database
    self._save_to_database(posts)
class InstaProcessManager:
    """
    Manager class for controlling Instagram monitoring processes.
    
    Attributes:
        db_host (str): MongoDB host address.
        db_port (int): MongoDB port number.
        processes (dict): Dictionary of running processes.
    """
    def __init__(
        self, 
        db_host=None,
        db_port=None
    ):
        """
        Initialize the process manager.
        
        Args:
            db_host (str, optional): MongoDB host. Defaults to config.host.
            db_port (int, optional): MongoDB port. Defaults to config.mongoPort.
        """
        self.db_host = db_host or config.host
        self.db_port = db_port or config.mongoPort
        self.processes = {}  # Track running processes

    def _get_process_db(self):
        """
        Get MongoDB collection for managing monitoring processes.
        
        Returns:
            pymongo.Collection: Processes collection.
        """
        client = pymongo.MongoClient(host=self.db_host, port=self.db_port)
        return client["insta_process"]["process"]

    def is_process_running(self, user, tags, product_id):
        """
        Check if a monitoring process is running.
        
        Args:
            user (str): User identifier.
            tags (str): Hashtag or profile name.
            product_id (str): Product identifier.
            
        Returns:
            bool: True if process is running, False otherwise.
        """
        try:
            collection = self._get_process_db()
            return collection.count_documents({
                "user": user,
                "tags": tags,
                "productId": product_id
            }) > 0
        except Exception as e:
            print(f"Error checking process status: {e}")
            return False

    def start_monitoring(self, user, tags, monitor_type="hashtags", product_id="insta_monitor"):
        """
        Start a new monitoring process.
        
        Args:
            user (str): User identifier.
            tags (str): Hashtag or profile name.
            monitor_type (str, optional): Monitoring type. Defaults to "hashtags".
            product_id (str, optional): Product identifier. Defaults to "insta_monitor".
            
        Returns:
            bool: True if process started successfully, False otherwise.
        """
        try:
            # Check if process is already running
            if self.is_process_running(user, tags, product_id):
                print(f"Monitoring for {user}/{tags} is already running")
                return False
                
            # Record process start in database
            collection = self._get_process_db()
            collection.insert_one({
                "user": user,
                "tags": tags,
                "monitorType": monitor_type,
                "productId": product_id,
                "startedAt": time.time()
            })
            
            # Start monitoring process
            process = multiprocessing.Process(
                target=self._run_monitoring_process,
                args=(user, tags, monitor_type, product_id)
            )
            process.start()
            
            # Track the process
            process_key = f"{user}_{tags}_{product_id}"
            self.processes[process_key] = process
            
            print(f"Started monitoring for {user}/{tags}")
            return True
            
        except Exception as e:
            print(f"Error starting monitoring: {e}")
            traceback.print_exc()
            return False

    def _run_monitoring_process(self, user, tags, monitor_type, product_id):
        """
        Run monitoring process in a separate process.
        
        Args:
            user (str): User identifier.
            tags (str): Hashtag or profile name.
            monitor_type (str): Monitoring type.
            product_id (str): Product identifier.
        """
        try:
            monitor = MonitoringClass(user, tags, monitor_type, product_id)
            
            # Continuously monitor until stopped
            while self.is_process_running(user, tags, product_id):
                monitor.run_monitoring()
                print(f"Waiting for next monitoring cycle for {user}/{tags}")
                time.sleep(300)  # 5-minute interval
                
            print(f"Monitoring stopped for {user}/{tags}")
            
        except Exception as e:
            print(f"Error in monitoring process: {e}")
            traceback.print_exc()

    def stop_monitoring(self, user, tags, product_id):
        """
        Stop a running monitoring process.
        
        Args:
            user (str): User identifier.
            tags (str): Hashtag or profile name.
            product_id (str): Product identifier.
            
        Returns:
            bool: True if process stopped successfully, False otherwise.
        """
        try:
            # Mark process as stopped in database
            collection = self._get_process_db()
            result = collection.delete_one({
                "user": user,
                "tags": tags,
                "productId": product_id
            })
            
            if result.deleted_count == 0:
                print(f"No running process found for {user}/{tags}")
                return False
                
            # Clean up process tracking
            process_key = f"{user}_{tags}_{product_id}"
            if process_key in self.processes:
                process = self.processes[process_key]
                process.terminate()
                process.join(timeout=5)
                del self.processes[process_key]
                
            print(f"Stopped monitoring for {user}/{tags}")
            return True
            
        except Exception as e:
            print(f"Error stopping monitoring: {e}")
            traceback.print_exc()
            return False

    def list_monitoring_processes(self):
        """
        List all running monitoring processes.
        
        Returns:
            list: List of running processes.
        """
        try:
            collection = self._get_process_db()
            return list(collection.find({}, {"_id": 0}))
        except Exception as e:
            print(f"Error listing processes: {e}")
            traceback.print_exc()
            return []


class DataFetcher:
    """
    Class for fetching Instagram data from MongoDB.
    
    Attributes:
        user (str): User identifier.
        tags (str): Hashtag or profile name.
        product_id (str): Product identifier.
        client (pymongo.MongoClient): MongoDB client.
        collection (pymongo.Collection): MongoDB collection.
    """
    def __init__(
        self, 
        user, 
        tags, 
        product_id="insta_monitor",
        db_host=None,
        db_port=None
    ):
        """
        Initialize the data fetcher.
        
        Args:
            user (str): User identifier.
            tags (str): Hashtag or profile name.
            product_id (str, optional): Product identifier. Defaults to "insta_monitor".
            db_host (str, optional): MongoDB host. Defaults to config.host.
            db_port (int, optional): MongoDB port. Defaults to config.mongoPort.
        """
        self.user = user
        self.tags = tags
        self.product_id = product_id
        self.db_host = db_host or config.host
        self.db_port = db_port or config.mongoPort
        self.client = None
        self.collection = None
        
        self._initialize()

    def _initialize(self):
        """Initialize MongoDB connection"""
        try:
            self.client = pymongo.MongoClient(host=self.db_host, port=self.db_port)
            db_name = f"{self.product_id}:{self.user}:insta"
            self.collection = self.client[db_name][self.tags]
        except Exception as e:
            print(f"Error initializing data fetcher: {e}")
            traceback.print_exc()
            raise

    def _cleanup(self):
        """Clean up resources (close MongoDB connection)"""
        if self.client:
            self.client.close()

    def get_recent_posts(self, limit=20):
        """
        Get recent Instagram posts.
        
        Args:
            limit (int, optional): Maximum number of posts to return. Defaults to 20.
            
        Returns:
            list: List of recent posts.
        """
        try:
            cursor = self.collection.find()\
                .sort("_inserted_at", pymongo.DESCENDING)\
                .limit(limit)
                
            return list(cursor)
        except Exception as e:
            print(f"Error fetching recent posts: {e}")
            traceback.print_exc()
            return []
        finally:
            self._cleanup()

    def get_posts_after_timestamp(self, timestamp, limit=20):
        """
        Get posts after a specific timestamp.
        
        Args:
            timestamp (int): Unix timestamp.
            limit (int, optional): Maximum number of posts to return. Defaults to 20.
            
        Returns:
            list: List of posts after the specified timestamp.
        """
        try:
            cursor = self.collection.find({"timestamp": {"$gt": timestamp}})\
                .sort("timestamp", pymongo.ASCENDING)\
                .limit(limit)
                
            return list(cursor)
        except Exception as e:
            print(f"Error fetching posts after timestamp: {e}")
            traceback.print_exc()
            return []
        finally:
            self._cleanup()

    def get_posts_before_timestamp(self, timestamp, limit=20):
        """
        Get posts before a specific timestamp.
        
        Args:
            timestamp (int): Unix timestamp.
            limit (int, optional): Maximum number of posts to return. Defaults to 20.
            
        Returns:
            list: List of posts before the specified timestamp.
        """
        try:
            cursor = self.collection.find({"timestamp": {"$lt": timestamp}})\
                .sort("timestamp", pymongo.DESCENDING)\
                .limit(limit)
                
            posts = list(cursor)
            posts.reverse()  # Reverse to maintain ascending order
            return posts
        except Exception as e:
            print(f"Error fetching posts before timestamp: {e}")
            traceback.print_exc()
            return []
        finally:
            self._cleanup()


def main():
    """
    Main function for starting Instagram monitoring from command line.
    
    Usage:
        python script.py [user] [tags] [monitor_type] [product_id]
        
    All arguments are optional and will use default values if not provided.
    """
    # Use command line arguments or default values
    user = sys.argv[1] if len(sys.argv) > 1 else "default_user"
    tags = sys.argv[2] if len(sys.argv) > 2 else "python"
    monitor_type = sys.argv[3] if len(sys.argv) > 3 else "hashtags"
    product_id = sys.argv[4] if len(sys.argv) > 4 else "insta_monitor"
    
    print("Starting Instagram monitor:")
    print(f"  User: {user}")
    print(f"  Tags: {tags}")
    print(f"  Type: {monitor_type}")
    print(f"  Product ID: {product_id}")
    
    try:
        manager = InstaProcessManager()
        manager.start_monitoring(user, tags, monitor_type, product_id)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()