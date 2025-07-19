import traceback
from concurrent.futures import ThreadPoolExecutor

import con_file
import tornado.ioloop
import tornado.web
import ujson
from tornado.concurrent import run_on_executor
from tornado.gen import coroutine

# Import required classes from local modules
try:
    # Attempt relative import (when running as part of a package)
    from .insta_datafetcher import DataFetcher, InstaProcessManager, MonitoringClass
    from .subpinsta import start_monitoring_process
except ImportError:
    # Fallback to absolute import (when running standalone)
    from insta_datafetcher import DataFetcher, InstaProcessManager
    from subpinsta import start_monitoring_process

# Configuration constants
MAX_WORKERS = 10  # Maximum number of threads for background tasks

class StartHandlerinsta(tornado.web.RequestHandler):
    """
    API handler to initiate Instagram monitoring for a specified hashtag or profile.
    
    Query Parameters:
        q (str): Hashtag or profile name to monitor
        userId (str): User identifier
        type (str): Monitoring type ('hashtags' or 'profile')
        productId (str): Product identifier for database storage
    """
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    @run_on_executor
    def background_task(self, user: str, tags: str, monitor_type: str, product_id: str) -> None:
        """
        Execute monitoring process in a background thread.
        
        Args:
            user: User identifier
            tags: Hashtag or profile name
            monitor_type: Type of monitoring ('hashtags' or 'profile')
            product_id: Product identifier for database
        """
        try:
            # Start monitoring process with configuration
            start_monitoring_process(
                user=user, 
                tags=tags, 
                monitor_type=monitor_type, 
                product_id=product_id,
                db_host=con_file.host,  # Use configured database host
                db_port=con_file.mongoPort  # Use configured database port
            )
        except Exception as e:
            print(f"Error starting monitoring process: {str(e)}")
            traceback.print_exc()

    @coroutine
    def get(self) -> None:
        """Handle GET request to start monitoring"""
        try:
            # Extract and validate query parameters
            q = self.get_argument("q")
            user = self.get_argument("userId")
            monitor_type = self.get_argument("type")
            product_id = self.get_argument("productId")
            
            # Sanitize input (remove spaces from tags)
            if " " in q:
                q = q.replace(" ", "")
                
            # Start background monitoring task
            self.background_task(user, tags=q, monitor_type=monitor_type, product_id=product_id)
            
            # Return success response
            response = {
                "query": q,
                "userId": user,
                "status": True,
                "productId": product_id,
                "message": f"Monitoring started for {monitor_type}: {q}"
            }
            
            print(f"Monitoring started: User={user}, Tags={q}, Type={monitor_type}")
            self.write(ujson.dumps(response))
            
        except Exception as e:
            # Handle missing parameters or other errors
            self.send_error(400, reason=f"Invalid request: {str(e)}")

class StopHandlerinsta(tornado.web.RequestHandler):
    """
    API handler to terminate Instagram monitoring for a specified hashtag or profile.
    
    Query Parameters:
        q (str): Hashtag or profile name to stop monitoring
        userId (str): User identifier
        productId (str): Product identifier
    """
    def get(self) -> None:
        """Handle GET request to stop monitoring"""
        try:
            # Extract and validate query parameters
            q = self.get_argument("q")
            user = self.get_argument("userId")
            product_id = self.get_argument("productId")
            
            # Stop monitoring process with configuration
            process_manager = InstaProcessManager(
                db_host=con_file.host,  # Use configured database host
                db_port=con_file.mongoPort  # Use configured database port
            )
            result = process_manager.stop_monitoring(user, q, product_id)
            
            # Return status response
            response = {
                "query": q,
                "userId": user,
                "productId": product_id,
                "status": result,
                "message": f"Monitoring stopped for {q}" if result else f"Monitoring not found for {q}"
            }
            
            print(f"Monitoring status: {q} - {'Stopped' if result else 'Not Found'}")
            self.write(ujson.dumps(response))
            
        except Exception as e:
            # Handle missing parameters or other errors
            self.send_error(400, reason=f"Invalid request: {str(e)}")

class StatusHandlerinsta(tornado.web.RequestHandler):
    """
    API handler to check the status of Instagram monitoring.
    
    Query Parameters:
        q (str): Hashtag or profile name
        userId (str): User identifier
        productId (str): Product identifier
    """
    def get(self) -> None:
        """Handle GET request to check monitoring status"""
        try:
            # Extract and validate query parameters
            q = self.get_argument("q")
            user = self.get_argument("userId")
            product_id = self.get_argument("productId")
            
            # Check monitoring status with configuration
            process_manager = InstaProcessManager(
                db_host=con_file.host,  # Use configured database host
                db_port=con_file.mongoPort  # Use configured database port
            )
            is_running = process_manager.is_process_running(user, q, product_id)
            
            # Return status response
            response = {
                "query": q,
                "userId": user,
                "status": is_running,
                "productId": product_id,
                "message": f"Monitoring is {'active' if is_running else 'inactive'} for {q}"
            }
            
            print(f"Status check: {q} - {'Active' if is_running else 'Inactive'}")
            self.write(ujson.dumps(response))
            
        except Exception as e:
            # Handle missing parameters or other errors
            self.send_error(400, reason=f"Invalid request: {str(e)}")

class SenderHandlerinstaLess(tornado.web.RequestHandler):
    """
    API handler to retrieve Instagram posts older than a specified timestamp.
    
    Query Parameters:
        q (str): Hashtag or profile name
        userId (str): User identifier
        type (str): Monitoring type
        productId (str): Product identifier
        date (int): Unix timestamp
        limit (int): Maximum number of posts to return
    """
    def get(self) -> None:
        """Handle GET request to fetch older posts"""
        try:
            # Extract and validate query parameters
            q = self.get_argument("q")
            user = self.get_argument("userId")
            monitor_type = self.get_argument("type")
            product_id = self.get_argument("productId")
            date = int(self.get_argument("date"))  # Unix timestamp
            limit = int(self.get_argument("limit"))  # Number of posts
            
            # Fetch posts older than specified timestamp with configuration
            data_fetcher = DataFetcher(
                user=user, 
                tags=q, 
                product_id=product_id,
                db_host=con_file.host,  # Use configured database host
                db_port=con_file.mongoPort  # Use configured database port
            )
            posts = data_fetcher.get_posts_before_timestamp(date, limit)
            
            # Return posts data
            self.write(ujson.dumps({
                "query": q,
                "userId": user,
                "productId": product_id,
                "count": len(posts),
                "posts": posts
            }))
            
        except ValueError as ve:
            # Handle invalid parameter types
            self.send_error(400, reason=f"Invalid parameter: {str(ve)}")
        except Exception as e:
            # Handle other errors
            self.send_error(500, reason=f"Internal server error: {str(e)}")

class SenderHandlerinstaGreater(tornado.web.RequestHandler):
    """
    API handler to retrieve Instagram posts newer than a specified timestamp.
    
    Query Parameters:
        q (str): Hashtag or profile name
        userId (str): User identifier
        type (str): Monitoring type
        productId (str): Product identifier
        date (int): Unix timestamp
        limit (int): Maximum number of posts to return
    """
    def get(self) -> None:
        """Handle GET request to fetch newer posts"""
        try:
            # Extract and validate query parameters
            q = self.get_argument("q")
            user = self.get_argument("userId")
            monitor_type = self.get_argument("type")
            product_id = self.get_argument("productId")
            date = int(self.get_argument("date"))  # Unix timestamp
            limit = int(self.get_argument("limit"))  # Number of posts
            
            # Fetch posts newer than specified timestamp with configuration
            data_fetcher = DataFetcher(
                user=user, 
                tags=q, 
                product_id=product_id,
                db_host=con_file.host,  # Use configured database host
                db_port=con_file.mongoPort  # Use configured database port
            )
            posts = data_fetcher.get_posts_after_timestamp(date, limit)
            
            # Return posts data
            self.write(ujson.dumps({
                "query": q,
                "userId": user,
                "productId": product_id,
                "count": len(posts),
                "posts": posts
            }))
            
        except ValueError as ve:
            # Handle invalid parameter types
            self.send_error(400, reason=f"Invalid parameter: {str(ve)}")
        except Exception as e:
            # Handle other errors
            self.send_error(500, reason=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    """Main entry point - Starts the Tornado server"""
    # Define API routes and handlers
    application = tornado.web.Application(
        [
            (r"/instagram/monitoring/start", StartHandlerinsta),
            (r"/instagram/monitoring/stop", StopHandlerinsta),
            (r"/instagram/monitoring/status", StatusHandlerinsta),
            (r"/instagram/monitoring/less", SenderHandlerinstaLess),
            (r"/instagram/monitoring/greater", SenderHandlerinstaGreater),
        ],
        debug=False,  # Disable debug mode for production
        autoreload=True  # Automatically reload on code changes
    )

    # Start the server
    port = 7074
    application.listen(port)
    print(f"Instagram Monitoring API Server running on port {port}")
    print(f"MongoDB connection: {con_file.host}:{con_file.mongoPort}")  # Display MongoDB connection info
    print("Available endpoints:")
    print("  - /instagram/monitoring/start")
    print("  - /instagram/monitoring/stop")
    print("  - /instagram/monitoring/status")
    print("  - /instagram/monitoring/less")
    print("  - /instagram/monitoring/greater")
    
    # Start the I/O loop
    tornado.ioloop.IOLoop.current().start()