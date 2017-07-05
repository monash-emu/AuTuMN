import datetime
from datetime import datetime
import traceback
from pprint import pformat

import dateutil
import dateutil.tz

from celery import Celery
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy.orm import scoped_session, sessionmaker

from .api import app

# WARNING: import only after app is defined
from . import dbmodel


# Concept: task_id: farms out async jobs
# generalize to obj_id irrespective of object type?

db = SQLAlchemy(app)

celery_instance = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
celery_instance.conf.update(app.config)

TaskBase = celery_instance.Task

class ContextTask(TaskBase):
    abstract = True

    def __call__(self, *args, **kwargs):
        with app.app_context():
            return TaskBase.__call__(self, *args, **kwargs)

celery_instance.Task = ContextTask


def init_scoped_db_session():
    return scoped_session(sessionmaker(db.engine))


def close_db_session(db_session):
    # this line might be redundant (not 100% sure - not clearly described)
    db_session.connection().close() # pylint: disable=E1101
    db_session.remove()
    # black magic to actually close the connection by forcing the engine to dispose of garbage (I assume)
    db_session.bind.dispose() # pylint: disable=E1101


def parse_tasklog(tasklog):
    return {
        'status': tasklog.status,
        'task_id': tasklog.task_id,
        'error_text': tasklog.error,
        'start_time': tasklog.start_time,
        'stop_time': tasklog.stop_time,
        'obj_id': tasklog.obj_id,
        'work_type': tasklog.work_type,
        'current_time': datetime.now(dateutil.tz.tzutc())
    }


def get_tasklog(task_id, db_session):
    query = db_session.query(dbmodel.TaskLogDb).filter_by(task_id=task_id)
    return query.first()


def setup_task(task_id):
    db_session = init_scoped_db_session()
    tasklog = get_tasklog(task_id, db_session)
    if tasklog is not None:
        if tasklog.status == 'started':
            status = parse_tasklog(tasklog)
            status["status"] = "blocked"
            print(">> setup_task: already exists similar job")
            return status
        tasklog.cleanup()
        db_session.delete(tasklog)

    tokens = task_id.split(":")
    work_type, obj_id = tokens[:2]
    print(">> setup_task '%s %s'" % (obj_id, work_type))

    tasklog = dbmodel.TaskLogDb(
        obj_id=obj_id,
        work_type=work_type,
        task_id=task_id,
        start_time=datetime.now(dateutil.tz.tzutc()))
    db_session.add(tasklog)
    db_session.flush()
    obj_str = dbmodel.load_obj_str(obj_id, "project", db_session)
    tasklog.save_obj_to_redis(obj_str)
    db_session.commit()
    status = parse_tasklog(tasklog)

    close_db_session(db_session)

    return status


def check_task(task_id):
    db_session = init_scoped_db_session()
    tasklog = get_tasklog(task_id, db_session)
    if tasklog:
        print(">> Found existing job of " + task_id)
        status = parse_tasklog(tasklog)
        print(pformat(status, indent=2))
        if status['status'] == 'error':
            tasklog.cleanup()
            db_session.delete(tasklog)
            db_session.commit()
            raise Exception(status['error_text'])
    else:
        status = {
            'status': 'unknown',
            'error_text': None,
            'start_time': None,
            'stop_time': None,
            'work_type': ''
        }
    close_db_session(db_session)
    return status


@celery_instance.task()
def run_task(task_id):
    db_session = init_scoped_db_session()
    tasklog = get_tasklog(task_id, db_session)
    if tasklog is not None:
        obj_str = tasklog.load_obj_from_redis()
    else:
        obj_str = None
    close_db_session(db_session)

    tokens = task_id.split(":")
    work_type, project_id = tokens[:2]
    task_args = tokens[2:]
    task_fn = globals()[work_type]
    try:
        if obj_str is None:
            raise ValueError("Work log not found")
        project = model.loads(obj_str)
        task_fn(project, *task_args)
        print("> tasks.run_task completed")
        error_text = ""
        status = 'completed'
    except Exception:
        error_text = traceback.format_exc()
        status = 'error'
        print(">> tasks.run_task error")
        print(error_text)

    db_session = init_scoped_db_session()
    if status == 'completed':
        dbmodel.save_object(
            project_id,
            "project",
            model.dumps(project),
            model.parse_project(project),
            db_session)
    tasklog = get_tasklog(task_id, db_session)
    tasklog.status = status
    tasklog.error = error_text
    tasklog.stop_time = datetime.now(dateutil.tz.tzutc())
    tasklog.cleanup()
    db_session.add(tasklog)
    db_session.commit()
    close_db_session(db_session)


# AVAILABLE TASKS

# def autofit(project, parset_id):
#     print("> tasks.autofit {project:%s, parset_id:%s}" % (project.name, parset_id))
#     orig_parset = model.get_parset(project, parset_id)
#     orig_parset_name = orig_parset.name
#     autofit_parset_name = "autofit-" + str(orig_parset_name)
#     project.runAutofitCalibration(
#         new_parset_name=autofit_parset_name,
#         old_parset_name=orig_parset_name)


