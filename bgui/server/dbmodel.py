from __future__ import print_function

import copy
from datetime import datetime
import dateutil.tz
import uuid
import json

from sqlalchemy.types import TypeDecorator, CHAR, VARCHAR
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import deferred

from conn import db


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    http://docs.sqlalchemy.org/en/latest/core/custom_types.html
    """
    impl = CHAR

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                print("> GUID.process_bind_param", value, type(value))
                # hexstring
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            return uuid.UUID(value)


class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string.

    Usage::

        JSONEncodedDict(255)

    """

    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class UserDb(db.Model):

    __tablename__ = 'users'

    id = db.Column(GUID(), default=uuid.uuid4, primary_key=True)
    username = db.Column(db.String(255))
    name = db.Column(db.String(60))
    email = db.Column(db.String(200))
    password = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, default=False)
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

    id = db.Column(GUID(), default=uuid.uuid4, nullable=False, unique=True, primary_key=True)
    user_id = db.Column(GUID(True), db.ForeignKey('users.id'))
    obj_type = db.Column(db.Text, default=None)
    attr = db.Column(JSONEncodedDict)
    blob = deferred(db.Column(db.LargeBinary))

    def load_obj_from_redis(self):
        print(">> ObjectDb.load_obj_from_redis " + self.id.hex)
        return self.blob

    def save_obj_to_redis(self, obj):
        print(">> ObjectDb.save_obj_to_redis " + self.id.hex)
        self.blob = obj

    def cleanup(self):
        print(">> ObjectDb.cleanup " + self.id.hex)
        self.blob = b''


# only needs to be done once but here just in case
db.create_all()


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
        'isAdmin': user.is_admin,
    }


def create_user(user_attr, db_session=None):
    db_session = verify_db_session(db_session)
    for key in user_attr:
        user_attr[key] = str(user_attr[key])
    user = UserDb(**user_attr)
    db_session.add(user)
    print(">> dbmodel.create_user", user_attr)
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


