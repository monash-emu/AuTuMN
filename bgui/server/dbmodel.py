from __future__ import print_function

import copy

import datetime
from datetime import datetime
import dateutil
import dateutil.tz

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import deferred

from .dbconn import db, redis


class UserDb(db.Model):

    __tablename__ = 'users'

    id = db.Column(UUID(True), server_default=text("uuid_generate_v1mc()"), primary_key=True)
    username = db.Column(db.String(255))
    name = db.Column(db.String(60))
    email = db.Column(db.String(200))
    password = db.Column(db.String(200))
    is_admin = db.Column(db.Boolean, server_default=text('FALSE'))
    objects = db.relationship('ObjectDb', backref='user', lazy='dynamic')

    def get_id(self):
        return self.id

    def is_active(self):  # pylint: disable=R0201
        return True

    def is_anonymous(self):  # pylint: disable=R0201
        return False

    def is_authenticated(self):  # pylint: disable=R0201
        return True


class ObjectDb(db.Model):

    __tablename__ = 'objects'

    id = db.Column(UUID(True), server_default=text("uuid_generate_v1mc()"), primary_key=True)
    user_id = db.Column(UUID(True), db.ForeignKey('users.id'))
    obj_type = db.Column(db.Text, default=None)
    owner_id = db.Column(UUID(True), db.ForeignKey('objects.id'), default=None) # rename to master_obj_id
    attr = db.Column(JSON)
    blob = deferred(db.Column(db.LargeBinary))

    def load_obj_from_redis(self):
        print(">> ObjectDb.load_obj_from_redis " + self.id.hex)
        return redis.get(self.id.hex)

    def save_obj_to_redis(self, obj):
        print(">> ObjectDb.save_obj_to_redis " + self.id.hex)
        redis.set(self.id.hex, obj)

    def cleanup(self):
        print(">> ObjectDb.cleanup " + self.id.hex)
        redis.delete(self.id.hex)



class TaskLogDb(db.Model):  # pylint: disable=R0903

    __tablename__ = "tasklogs"

    id = db.Column(UUID(True), server_default=text("uuid_generate_v1mc()"), primary_key=True)
    work_type = db.Column(db.String(128), default=None)
    task_id = db.Column(db.String(128), default=None)
    obj_id = db.Column(UUID(True))
    start_time = db.Column(db.DateTime(timezone=True), server_default=text('now()'))
    stop_time = db.Column(db.DateTime(timezone=True), default=None)
    work_status = db.Enum('started', 'completed', 'cancelled', 'error', 'blocked', name='work_status')
    status = db.Column(work_status, default='started')
    error = db.Column(db.Text, default=None)

    def load_obj_from_redis(self):
        print(">> TaskLogDb.load_obj_from_redis task-" + self.id.hex)
        return redis.get("task-" + self.id.hex)

    def save_obj_to_redis(self, obj):
        print(">> TaskLogDb.save_obj_to_redis task-" + self.id.hex)
        redis.set("task-" + self.id.hex, obj)

    def cleanup(self):
        print(">> TaskLogDb.cleanup task-" + self.id.hex)
        redis.delete("task-" + self.id.hex)


def filter_dict_for_none(d):
    new_d = {}
    for key, value in d.items():
        if value is not None:
            new_d[key] = value
    return new_d


def verify_db_session(db_session=None):
    if db_session is None:
        return db.session
    return db_session


# OBJECT functions

def make_obj_query(user_id=None, obj_type="project", db_session=None, **kwargs):
    db_session = verify_db_session(db_session)
    kwargs = filter_dict_for_none(kwargs)
    if user_id is not None:
        kwargs['user_id'] = user_id
    if obj_type is not None:
        kwargs['obj_type'] = obj_type
    print(">> dbmodel.make_obj_query", kwargs)
    return db_session.query(ObjectDb).filter_by(**kwargs)


def load_obj_attr(id=id, obj_type="project", db_session=None):
    query = make_obj_query(id=id, obj_type=obj_type, db_session=db_session)
    return query.one().attr


def load_obj_records(user_id=None, obj_type="project", db_session=None):
    query = make_obj_query(user_id=user_id, obj_type=obj_type, db_session=db_session)
    return query.all()


def load_obj_attr_list(user_id=None, obj_type="project", db_session=None):
    records = load_obj_records(user_id=user_id, obj_type=obj_type, db_session=db_session)
    return [record.attr for record in records]


def create_obj_id(db_session=None, **kwargs):
    db_session = verify_db_session(db_session)
    record = ObjectDb(**kwargs)
    db_session.add(record)
    db_session.commit()
    return record.id


def save_object(id, obj_type, obj_str, obj_attr, db_session=None):
    db_session = verify_db_session(db_session)
    record = make_obj_query(id=id, obj_type=obj_type, db_session=db_session).one()
    record.save_obj_to_redis(obj_str)
    obj_attr = copy.deepcopy(obj_attr)
    obj_attr['userId'] = str(record.user_id)
    obj_attr['modifiedTime'] = repr(datetime.now(dateutil.tz.tzutc()))
    record.attr = obj_attr
    db_session.add(record)
    db_session.commit()


def get_user_id(obj_id, db_session=None):
    record = make_obj_query(id=obj_id, db_session=db_session).one()
    return record.user_id


def load_obj_str(obj_id, obj_type, db_session=None):
    record = make_obj_query(id=obj_id, obj_type=obj_type, db_session=db_session).one()
    return record.load_obj_from_redis()


def delete_obj(obj_id, db_session=None):
    db_session = verify_db_session(db_session)
    record = make_obj_query(id=obj_id, db_session=db_session).one()
    db_session.delete(record)
    db_session.commit()


# USER functions

def parse_user(user):
    return {
        'id': user.id,
        'name': user.name,
        'username': user.username,
        'email': user.email,
        'is_admin': user.is_admin,
    }


def create_user(user_attr, db_session=None):
    db_session = verify_db_session(db_session)
    print(">> create_user", user_attr)
    user = UserDb(**user_attr)
    db_session.add(user)
    db_session.commit()
    return parse_user(user)


def make_user_query(db_session=None, **kwargs):
    db_session = verify_db_session(db_session)
    kwargs = filter_dict_for_none(kwargs)
    return db_session.query(UserDb).filter_by(**kwargs)


def load_user(db_session=None, **kwargs):
    query = make_user_query(db_session=db_session, **kwargs)
    return query.one()


def load_users():
    return make_user_query().all()


def update_user_from_attr(user_attr, db_session=None):
    db_session = verify_db_session(db_session)
    user = load_user(id=user_attr['id'])
    for key, value in user_attr.items():
        if value is not None:
            setattr(user, key, value)
    db_session.add(user)
    db_session.commit()
    return parse_user(user)


def delete_user(user_id, db_session=None):
    db_session = verify_db_session(db_session)
    user = load_user(id=user_id)
    user_attr = parse_user(user)
    query = make_obj_query(user_id=user_id, db_session=db_session)
    for record in query.all():
        db_session.delete(record)
    db_session.delete(user)
    db_session.commit()
    return user_attr


# tasklog functions

def cleanup_tasklogs(db_session):
    tasklogs = db_session.query(TaskLogDb)
    n = tasklogs.count()
    if n:
        print("> dbmodel.init clear {} worklogs".format(n))
        for work_log in tasklogs:
            work_log.cleanup()
        tasklogs.delete()
        db_session.commit()






