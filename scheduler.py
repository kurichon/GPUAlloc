from pytz import utc

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ProcessPoolExecutor


job_defaults = {
    'coalesce': False,
    'max_instances': 3
}
scheduler = BackgroundScheduler()

def myfunc():
    print("my_job_id is invoked")
scheduler.add_job(myfunc, 'interval',seconds=5, id='my_job_id')



scheduler.print_jobs()
# .. do something else here, maybe add jobs etc.

scheduler.configure(job_defaults=job_defaults, timezone=utc)
scheduler.start()
