import enum
import os
import sqlalchemy
import sqlalchemy.orm
from sqlalchemy import Column, Enum, Float, ForeignKey, Index, Integer, String, Table
from sqlalchemy import distinct, not_
from sqlalchemy.schema import UniqueConstraint


class ModelType(enum.Enum):
    IMAGE = "image"
    POSTCARD = "postcard"


class Base(sqlalchemy.orm.DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "image"

    id = Column(Integer, primary_key=True)
    origin = Column(String(64), nullable=False)
    path = Column(String(1024), nullable=False)
    size = Column(Integer)
    sha256 = Column(String(64))
    md5 = Column(String(32))
    height = Column(Integer)
    width = Column(Integer)

    __table_args__ = (
        UniqueConstraint('origin', 'path'),
        Index('idx_image_sha256', 'sha256'),
    )


class Postcard(Base):
    __tablename__ = "postcard"

    front_sha256 = Column(String(64), ForeignKey("image.sha256"), unique=True)
    back_sha256 = Column(String(64), ForeignKey("image.sha256"), primary_key=True)


class Model(Base):
    __tablename__ = "model"

    name = Column(String(64), primary_key=True)
    type = Column(Enum(ModelType), nullable=False)
    python_class = Column(String(64))
    labels = Column(String(1024)) # This should be a JSON list


class Label(Base):
    __tablename__ = "label"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(64), ForeignKey("model.name"))
    # Note that the sha256 below might refer to:
    # - image.sha256 (when the model type is IMAGE)
    # - postcard.back_sha256 (when the model type is POSTCARD)
    sha256 = Column(String(64), ForeignKey("image.sha256"))
    prediction = Column(String(256))
    label = Column(String(16))

    __table_args__ = (
        UniqueConstraint('model_name', 'sha256'),
    )


engine = sqlalchemy.create_engine("sqlite:///images.db")
Session = sqlalchemy.orm.sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def get_root(origin):
    return f"/mnt/FMRA-{origin}"


def make_path(origin, path):
    return os.path.join(get_root(origin), path)
