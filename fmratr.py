#!/usr/bin/env python
import click
import db
import hashlib
import json
import logging
import os
import PIL.Image
from tqdm import tqdm


logging.basicConfig()
log = logging


@click.group()
def cli():
    pass


def insert_file(origin, relative_path, absolute_path, session):
    # Check if the file already exists in the table
    existing_image = (
        session.query(db.Image)
        .filter_by(origin=origin, path=relative_path)
        .first()
    )
    if existing_image:
        return                    
    size = os.stat(absolute_path).st_size
    new_image = db.Image(origin=origin, path=relative_path, size=size)
    session.add(new_image)


@cli.command()
@click.argument("origin")
@click.argument("model")
@click.argument("label")
@click.argument("file", type=click.File())
def import_label_from_list(origin, model, label, file):
    with db.Session() as session:
        model = session.query(db.Model).filter_by(name=model).one()
        labels = json.loads(model.labels)
        assert label in labels
        for line in file:
            path = line.rstrip("\n")
            image = session.query(db.Image).filter_by(origin=origin, path=path).one()
            existing_label = session.query(db.Label).filter_by(model_name=model.name, sha256=image.sha256).first()
            if existing_label:
                existing_label.label = label
            else:
                new_label = db.Label(model_name=model.name, sha256=image.sha256, label=label)
                session.add(new_label)
        session.commit()   


@cli.command()
@click.argument("origin")
@click.argument("file", type=click.File())
def import_from_list(origin, file):
    origin_root = db.get_root(origin)
    assert os.path.isdir(origin_root), "Couldn't find that origin's root directory."
    with db.Session() as session:
        for line in tqdm(file):
            relative_path = line.rstrip("\n")
            absolute_path = db.make_path(origin, relative_path)
            insert_file(origin, relative_path, absolute_path, session)
        session.commit()   


@cli.command()
@click.argument("origin")
def verify(origin):
    origin_root = db.get_root(origin)
    assert os.path.isdir(origin_root), "Couldn't find that origin's root directory."
    with db.Session() as session:
        q = session.query(db.Image).filter_by(origin=origin)
        for image in tqdm(q, total=q.count()):
            absolute_path = db.make_path(image.origin, image.path)
            if not os.path.isfile(absolute_path):
                print(f"⚠️ File not found: {image.path}")
                continue
            if not image.sha256:
                print(f"⚠️ No SHA256 in database: {image.path}")
                continue
            sha256 = hashlib.sha256(open(absolute_path, "rb").read()).hexdigest()
            if sha256 != image.sha256:
                print(f"⚠️ SHA256 mismatch: {image.path}")


@cli.command()
@click.argument("origin")
@click.option("-l", "--limit", "limit", default=0)
def import_from_directory(origin, limit):
    origin_root = db.get_root(origin)
    assert os.path.isdir(origin_root), "Couldn't find that origin's root directory."
    with tqdm(total=limit) as progress_bar:
        with db.Session() as session:
            count = 0
            for root, dirs, files in os.walk(origin_root):
                if count > limit > 0:
                    break
                for file_name in files:
                    progress_bar.update()
                    count += 1
                    absolute_path = os.path.join(root, file_name)
                    relative_path = absolute_path[len(origin_root):].lstrip("/")
                    insert_file(origin, relative_path, absolute_path, session)
                # For performance reasons, don't commit after every single file.
                # Commit after a whole directory instead.
                session.commit()


@cli.command()
def add_image_metadata():
    commit_after = 1000
    with db.Session() as session:
        images_without_metadata = session.query(db.Image).filter_by(sha256=None)
        for i, image in tqdm(enumerate(images_without_metadata), total=images_without_metadata.count()):
            image_path = db.make_path(image.origin, image.path)
            image.sha256 = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
            image.md5 = hashlib.md5(open(image_path, "rb").read()).hexdigest()
            try:
                image.width, image.height = PIL.Image.open(image_path).size
            except Exception as e:
                log.error(e)
            if i % commit_after == 0:
                session.commit()
        session.commit()


@cli.command()
@click.argument("model")
def train(model):
    import ml
    ml.train(model)


@cli.command()
@click.argument("model")
def predict(model):
    import ml
    ml.inference(model, mode="predict")


@cli.command()
@click.argument("model")
def analyze(model):
    import ml
    ml.inference(model, mode="analyze")


@cli.command()
def webui():
    import ui
    ui.app.run(host="0.0.0.0")


if __name__ == "__main__":
    cli()
