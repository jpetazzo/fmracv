#!/usr/bin/env python
import click
import flask
import json
import logging
import os
import pickle
import random
import requests
import tensorflow as tf
from tqdm import tqdm
import yaml


BASE_DIR = os.environ.get("BASE_DIR", "imgroot")
BATCH_SIZE = 32
EPOCHS = 10
CHANNELS = 3
IMAGE_WIDTH, IMAGE_HEIGHT = (224, 224)
DTYPE = tf.float16
VRAM = 10_000_000_000
CONFIDENCE_THRESHOLD = 0.95


class Model:
    def __init__(self, config_file):
        self.config = yaml.safe_load(open(config_file))
        self.model_file = self.config["model_file"]
        self.labels = []
        for category_index, category_dict in enumerate(self.config["training_data"]):
            assert len(category_dict) == 1
            label, image_lists = list(category_dict.items())[0]
            self.labels.append(label)

    def load_image_lists(self):
        self.images = []
        for category_index, category_dict in enumerate(self.config["training_data"]):
            assert len(category_dict) == 1
            label, image_lists = category_dict.popitem()
            for image_list in image_lists:
                for image_file in open(image_list).read().strip().split("\n"):
                    self.images.append(dict(filename=image_file, label=category_index))

    def shuffle_image_list(self):
        random.shuffle(self.images)

    def truncate_image_list(self):
        image_count = VRAM // IMAGE_WIDTH // IMAGE_HEIGHT // CHANNELS // DTYPE.size
        print(f"Truncating data set to keep only {image_count} images.")
        del self.images[image_count:]

    def split_data(self):
        training_size = int(0.8 * len(self.images))
        self.training_images = self.images[:training_size]
        self.validation_images = self.images[training_size:]

    def load_images(self):
        for image_dict in tqdm(self.images):
            if "tensor" not in image_dict:
                tensor = load_image(image_dict["filename"])
                if tensor is not None:
                    image_dict["tensor"] = tensor

    def _build_big_tensors(self, images):
        tensors = []
        labels = []
        for image in images:
            if "tensor" not in image:
                continue
            tensors.append(image["tensor"])
            labels.append(image["label"])
        return (
            tf.stack(tensors),
            tf.keras.utils.to_categorical(labels, num_classes=len(self.labels)),
        )

    def build_big_tensors(self):
        self.training_x, self.training_y = self._build_big_tensors(self.training_images)
        self.validation_x, self.validation_y = self._build_big_tensors(
            self.validation_images
        )

    def build_model(self):
        # Okay we don't really know why this particular model architecture is better, but... here we go
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(len(self.labels), activation="softmax"),
            ]
        )

    def train_model(self):
        # Compile the model with:
        # - categorical crossentropy loss (because we can have more than 2 classes?)
        # - Adam optimizer (but we should also try Adagrad to see if it performs better)
        # - at the end of each epoch, show the accuracy compared to the validation set
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        # Train the model
        self.model.fit(
            self.training_x,
            self.training_y,
            validation_data=(self.validation_x, self.validation_y),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )
        self.model.save(self.model_file)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_file)

    def predict_batch_images(self, images):
        with tf.device("/CPU:0"):
            for image in images:
                if "tensor" not in image:
                    image["tensor"] = load_image(image["filename"])
            # Don't include images that failed to load
            images = [ image for image in images if image["tensor"] is not None ]
            if not images:
                raise ValueError("no image had a valid tensor")
            batch_tensor = tf.stack([image["tensor"] for image in images])
        batch_predictions = self.model.predict(batch_tensor, verbose=0)
        for image, prediction in zip(images, batch_predictions):
            image["prediction"] = {}
            for label_index, label_text in enumerate(self.labels):
                image["prediction"][label_text] = round(float(prediction[label_index]), 3)


def load_image(file_name):
    pickle_file = os.path.join(BASE_DIR, file_name) + ".pck"
    if os.path.isfile(pickle_file):
        return pickle.load(open(pickle_file, "rb"))
    try:
        data = tf.io.read_file(os.path.join(BASE_DIR, file_name))
        return load_image_from_bytes(data)
    except Exception as e:
        print(f"⚠️ Failed to load image {file_name} ({e})")
        return


def load_image_from_bytes(data, file_name="<no filename available>"):
    img = tf.image.decode_jpeg(data, channels=CHANNELS)
    height, width, channels = img.shape
    if height > width:
        img = tf.image.rot90(img)
        height, width, channels = img.shape
    ratio = width / height
    if ratio > 2:
        raise ValueError(f"Image has width/height ratio greater than 2 ({width}x{height})")
    img = tf.image.resize_with_pad(
        img, target_width=IMAGE_WIDTH, target_height=IMAGE_HEIGHT
    )
    # Normalize pixel values to [0,1]
    img = img / 255.0
    return tf.cast(img, DTYPE)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_file")
def train(config_file):
    m = Model(config_file)
    with tf.device("/CPU:0"):
        m.load_image_lists()
        m.shuffle_image_list()
        #m.truncate_image_list()
        m.split_data()
        m.load_images()
        m.build_big_tensors()
    print(tf.config.experimental.get_memory_info("GPU:0"))
    m.build_model()
    print(tf.config.experimental.get_memory_info("GPU:0"))
    m.train_model()
    print(tf.config.experimental.get_memory_info("GPU:0"))


@cli.command()
@click.argument("config_file")
def serve(config_file):
    m = Model(config_file)
    m.load_model()
    app = flask.Flask(__name__)
    @app.errorhandler(Exception)
    def handle_exception(error):
        logging.error(error, exc_info=True)
        response = flask.jsonify({
            "error": error.__class__.__name__,
            "details": str(error)
        })
        response.status_code = 500
        return response
    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if "image" in flask.request.files:
            data = flask.request.files["image"].read()
            tensor = load_image_from_bytes(data)
            image = dict(tensor=tensor)
            m.predict_batch_images([image])
            return flask.jsonify(image["prediction"])
        urls = flask.request.args.get("urls") or flask.request.form.get("urls")
        if urls:
            urls = json.loads(urls)
            images = []
            for url in urls:
                image = dict(url=url, tensor=None)
                images.append(image)
                try:
                    image["tensor"] = load_image_from_bytes(requests.get(url).content)
                except Exception as e:
                    image["prediction"] = dict(
                        error = e.__class__.__name__,
                        details = str(e)
                    )
            m.predict_batch_images(images)
            return flask.jsonify({
                image["url"]: image["prediction"]
                for image in images
            })
        return "This endpoint expects either an 'image' or a list of 'urls'.", 400

    @app.route("/", methods=["GET"])
    def index():
        return f"Model configuration: {config_file}\n"
    app.run(host="0.0.0.0", port=m.config["port"])


@cli.command()
@click.argument("config_file")
def analyze(config_file):
    m = Model(config_file)
    m.load_model()
    m.load_image_lists()
    output = []
    batch_size = 128
    for batch_index in tqdm(range(0, len(m.images), batch_size)):
        batch = m.images[batch_index:batch_index+batch_size]
        m.predict_batch_images(batch)
        for image in batch:
            if image["prediction"][m.labels[image["label"]]] < CONFIDENCE_THRESHOLD:
                # boo, we either got it wrong, or we're not sure enough
                data = dict(file_name=image["filename"], label=m.labels[image["label"]])
                data.update(image["prediction"])
                output.append(data)
    with open("inference-analysis.json", "w") as f:
        json.dump(output, f)


@cli.command()
@click.argument("config_file")
@click.argument("image_list_file")
def predict(config_file, image_list_file):
    m = Model(config_file)
    m.load_model()
    image_list = open(image_list_file).read().strip().split("\n")
    images = [ dict(filename=image_file) for image_file in image_list ]
    output_lists = {}
    output_all = []
    output_inconclusive = []
    batch_size = 128
    for batch_index in tqdm(range(0, len(images), batch_size)):
        batch = images[batch_index:batch_index+batch_size]
        m.predict_batch_images(batch)
        prediction = "inconclusive"
        for image in batch:
            if "prediction" not in image:
                image["prediction"] = {}
            for label, confidence in image["prediction"].items():
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction = label
            if prediction not in output_lists:
                output_lists[prediction] = open(image_list_file + "." + prediction, "w")
            output_lists[prediction].write(image["filename"] + "\n")
            data = dict(file_name=image["filename"])
            data.update(image["prediction"])
            output_all.append(data)
            if prediction == "inconclusive":
                output_inconclusive.append(data)
    with open("inference-all.json", "w") as f:
        json.dump(output_all, f)
    with open("inference-inconclusive.json", "w") as f:
        json.dump(output_inconclusive, f)


if __name__ == "__main__":
    cli()
