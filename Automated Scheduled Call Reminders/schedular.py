"""
Firebase-Twilio Reminder Scheduler

This script schedules the `search` function from the `caller` module to run at regular intervals.
The scheduler checks for upcoming reminders every hour and initiates calls 5 minutes before each scheduled time.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from caller import search

# Initialize the blocking scheduler
scheduler: BlockingScheduler = BlockingScheduler()


def start_scheduler() -> None:
    """
    Start the blocking scheduler to run the reminder search function at regular intervals.
    
    This function configures the scheduler to call the `search` function every hour
    and starts the scheduler in blocking mode. The process will run indefinitely
    until explicitly terminated.
    
    Raises:
        Any exceptions raised by the underlying scheduler implementation.
    """
    try:
        # Schedule the search function to run every hour
        scheduler.add_job(
            func=search,              # Function to call
            trigger="interval",       # Interval-based triggering
            hours=1,                  # Interval duration (1 hour)
            id="reminder_search_job", # Unique job identifier
            max_instances=1           # Ensure only one instance runs at a time
        )
        
        print("Scheduler initialized. Next run will be in 1 hour. Press Ctrl+C to exit.")
        
        # Start the scheduler in blocking mode (this thread will be occupied)
        scheduler.start()
        
    except KeyboardInterrupt:
        print("\nScheduler terminated by user.")
    except Exception as e:
        print(f"Error starting scheduler: {str(e)}")
        raise


if __name__ == "__main__":
    # Start the scheduler when run as a standalone script
    start_scheduler()