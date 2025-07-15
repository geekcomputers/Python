# schedular code for blocking schedular as we have only 1 process to run

from apscheduler.schedulers.blocking import BlockingScheduler

from caller import search


sched = BlockingScheduler()

# Schedule job_function to be called every two hours
sched.add_job(search, "interval", hours=1)  # for testing instead add hours =1

sched.start()
