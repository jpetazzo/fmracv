import db
import json
import os
import pickle
import PIL
import PIL.ImageFile
import random
import tensorflow as tf
import tensorflow.keras.layers as tkl
from tqdm import tqdm


BATCH_SIZE = 16
EPOCHS = 10
CHANNELS = 3
IMAGE_WIDTH, IMAGE_HEIGHT = (224, 224)
DTYPE = tf.float16
CONFIDENCE_THRESHOLD = 0.95
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")
TRAINING_VALIDATION_SPLIT = 0.9
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class CacheLoader:

    def __init__(self, load_function, cache_prefix):
        self.load_function = load_function
        self.cache_prefix = cache_prefix

    def load(self, absolute_path, sha256):
        pickle_file = os.path.join(CACHE_DIR, f"{self.cache_prefix}.{sha256}.pck")
        if os.path.isfile(pickle_file):
            return pickle.load(open(pickle_file, "rb"))
        else:
            tensor = self.load_function(absolute_path)
            pickle.dump(tensor, open(pickle_file, "wb"))
            return tensor


class ImageClassifierType1(tf.keras.models.Sequential):

    def __init__(self, labels):
        self.labels = labels
        super().__init__([
            tkl.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)),
            tkl.Conv2D(32, (3, 3), activation="relu"),
            tkl.MaxPooling2D((2, 2)),
            tkl.Conv2D(64, (3, 3), activation="relu"),
            tkl.MaxPooling2D((2, 2)),
            tkl.Conv2D(128, (3, 3), activation="relu"),
            tkl.MaxPooling2D((2, 2)),
            tkl.Conv2D(256, (3, 3), activation="relu"),
            tkl.MaxPooling2D((2, 2)),
            tkl.Flatten(),
            tkl.Dense(256, activation="relu"),
            tkl.Dense(256, activation="relu"),
            tkl.Dense(len(self.labels), activation="softmax"),
        ])


def load_as_landscape(filename_or_fileobj):
    i = PIL.Image.open(filename_or_fileobj)
    # If the image is in grayscale, convert it to RGB
    # (since our model is trained on RGB images)
    if i.mode == "L":
        i = i.convert("RGB")
    img = tf.convert_to_tensor(i)
    height, width, channels = img.shape
    assert channels == CHANNELS
    if height > width:
        img = tf.image.rot90(img)
        height, width, channels = img.shape
    img = tf.image.resize_with_pad(
        img, target_width=IMAGE_WIDTH, target_height=IMAGE_HEIGHT
    )
    # Normalize pixel values to [0,1]
    img = img / 255.0
    return tf.cast(img, DTYPE)


def meh():


    def arrange_front_and_back(self):
        # When processing pairs of images, make sure that we always have (front,back)
        # (and not the other way around).
        image_lists = set(i["list"] for i in self.images)
        fronts = set()
        backs = set()
        for image_list in image_lists:
            with open(image_list + ".front") as f:
                for line in f:
                    fronts.add(line.strip())
            with open(image_list + ".back") as f:
                for line in f:
                    backs.add(line.strip())
        for image in self.images:
            i0, i1 = image["filename"]
            if i0 in fronts and i1 in backs:
                continue
            elif i0 in backs and i1 in fronts:
                image["filename"] = i1, i0
            else:
                print("⚠️ Problem with front/back detection:")
                print(repr(i0))
                print(f"+++ front:{i0 in fronts} back:{i0 in backs}")
                print(repr(i1))
                print(f"+++ front:{i1 in fronts} back:{i1 in backs}")
                image["filename"] = "",""



    def build_model_PAIR_TRUE(self):

        x = input_image_pair = tkl.Input(shape=(2, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))

        x = tkl.Lambda(lambda x: x[:, 0, :, :])(input_image_pair)
        x = tkl.Conv2D(32, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Conv2D(64, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Conv2D(128, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Conv2D(256, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Flatten()(x)
        x = tkl.Dense(256, activation="relu")(x)
        cnn1 = x

        x = tkl.Lambda(lambda x: x[:, 1, :, :])(input_image_pair)
        x = tkl.Conv2D(32, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Conv2D(64, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Conv2D(128, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Conv2D(256, (3, 3), activation="relu")(x)
        x = tkl.MaxPooling2D((2, 2))(x)
        x = tkl.Flatten()(x)
        x = tkl.Dense(256, activation="relu")(x)
        cnn2 = x

        x = tkl.Concatenate()([cnn1, cnn2])
        #x = tkl.Dense(256, activation="relu")(x)
        x = output = tkl.Dense(len(self.labels), activation="softmax")(x)

        self.model = tf.keras.Model(inputs=[input_image_pair], outputs=[output])


def make_tensors(images, labels):
    image_tensor_list = []
    label_list = []
    for image, label in images:
        image_tensor_list.append(image)
        label_list.append(labels.index(label))
    return (
        tf.stack(image_tensor_list),
        tf.keras.utils.to_categorical(label_list, num_classes=len(labels)),
    )


def load(model_name):
    return tf.keras.models.load_model(os.path.join(SNAPSHOTS_DIR, model_name))


# "mode" can be:
# - "predict" (compute predictions for the first time)
# - "analyze" (re-compute predictions for everything, including labeled data)
def inference(model_name, mode):
    assert mode in ("predict", "analyze")
    model_instance = load(model_name)
    batch_size = 128
    loader = CacheLoader(load_as_landscape, "LAL224")
    with db.Session() as session:
        model_db = session.query(db.Model).filter_by(name=model_name).one()
        model_labels = json.loads(model_db.labels)
        done = session.query(db.Label.sha256).filter_by(model_name=model_name)
        if mode == "predict":
            todo = session.query(db.Image).filter(db.not_(db.Image.sha256.in_(done))).filter(db.Image.width > 0)
        if mode == "analyze":
            todo = session.query(db.Image).filter(db.Image.sha256.in_(done))
        todo = todo.all()
        ##### BEGIN HACK
        # For the with-address-or-blank model, we only want to consider the back of the postcard.
        if model_name == "with-address-or-blank":
            front_shas = set([l[0] for l in session.query(db.Label.sha256).filter_by(model_name="front-vs-back", label="back").all()])
            todo = [ i for i in todo if i.sha256 in front_shas ]
        ##### END HACK
        # FIXME: we really need to find a way to avoid these duplicates; that's annoying!
        todo_dedup = []
        todo_shas = set()
        for image in todo:
            if image.sha256 in todo_shas:
                continue
            todo_shas.add(image.sha256)
            todo_dedup.append(image)
        todo = todo_dedup
        for batch_index in tqdm(range(0, len(todo), batch_size)):
            tensors = []
            batch = todo[batch_index:batch_index+batch_size]
            for image in batch:
                absolute_path = db.make_path(image.origin, image.path)
                tensor = loader.load(absolute_path, image.sha256)
                tensors.append(tensor)
            batch_tensor = tf.stack(tensors)
            batch_predictions = model_instance.predict(batch_tensor, verbose=0)
            for image, prediction in zip(batch, batch_predictions):
                prediction_dict = {}
                for label_index, label_text in enumerate(model_labels):
                    prediction_dict[label_text] = round(float(prediction[label_index]), 3)
                prediction_json = json.dumps(prediction_dict)
                if mode == "predict":
                    session.add(db.Label(
                        model_name=model_name,
                        sha256=image.sha256,
                        prediction=prediction_json
                    ))
                if mode == "analyze":
                    q = session.query(db.Label).filter_by(model_name=model_name, sha256=image.sha256)
                    q.update({"prediction": prediction_json})
            session.commit()


def train(model_name):
    loader = CacheLoader(load_as_landscape, "LAL224")
    with db.Session() as session:
        model_db = session.query(db.Model).filter_by(name=model_name).one()
        model_class = globals()[model_db.python_class]
        model_labels = json.loads(model_db.labels)
        model_instance = model_class(model_labels)
        #model_instance.summary()

        # Load the data to CPU memory, because if there are too many images,
        # we won't have enough GPU memory to load them all.
        with tf.device("/CPU:0"):
            training_data = []
            validation_data = []
            for label in model_labels:
                # The following can probably be improved by writing better SQLAlchemy!
                # We're basically retrieving all the images for which a label has been
                # set, but the labels are associated to SHA256 values, we don't want
                # duplicates. So retrieve all images, then remove duplicates by checking
                # their SHA256 values.
                images = (
                    session.query(db.Image)
                    .join(db.Label)
                    .filter(db.Label.model_name==model_name, db.Label.label==label)
                )
                list_of_sha256_origin_path = []
                set_of_sha256 = set()
                for image in images:
                    if image.sha256 not in set_of_sha256:
                        list_of_sha256_origin_path.append((image.sha256, image.origin, image.path))
                        set_of_sha256.add(image.sha256)
                random.shuffle(list_of_sha256_origin_path)
                # Hack to speed up training time during testing.
                #list_of_sha256_origin_path = list_of_sha256_origin_path[:1000]
                training_data_size = int(TRAINING_VALIDATION_SPLIT*len(list_of_sha256_origin_path))
                for image_list, data in (
                    (list_of_sha256_origin_path[:training_data_size], training_data),
                    (list_of_sha256_origin_path[training_data_size:], validation_data),
                ):
                    for sha256, origin, path in tqdm(image_list):
                        absolute_path = db.make_path(origin, path)
                        tensor = loader.load(absolute_path, sha256)
                        data.append((tensor, label))
            random.shuffle(training_data)
            training_x, training_y = make_tensors(training_data, model_labels)
            validation_x, validation_y = make_tensors(validation_data, model_labels)

        # From that point, we will be working with GPU memory.
        # It looks like Tensorflow will automatically pull tensors from CPU
        # to GPU memory as needed. (?)

        # Compile the model with:
        # - categorical crossentropy loss (because we can have more than 2 classes?)
        # - Adam optimizer (but we should also try Adagrad to see if it performs better)
        # - at the end of each epoch, show the accuracy compared to the validation set
        model_instance.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        # Train the model
        model_instance.fit(
            training_x,
            training_y,
            validation_data=(validation_x, validation_y),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
        )
        model_instance.save(os.path.join(SNAPSHOTS_DIR, model_name))


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


def analyze(config_file):
    m = Model(config_file)
    m.load_model()
    m.load_image_lists()
    m.arrange_front_and_back()
    output = []
    batch_size = 128
    for batch_index in tqdm(range(0, len(m.images), batch_size)):
        batch = m.images[batch_index:batch_index+batch_size]
        m.predict_batch_images(batch)
        for image in batch:
            if "prediction" not in image:
                continue
            if image["prediction"][m.labels[image["label"]]] < CONFIDENCE_THRESHOLD:
                # boo, we either got it wrong, or we're not sure enough
                data = dict(file_name=image["filename"], label=m.labels[image["label"]])
                data.update(image["prediction"])
                output.append(data)
    with open("inference-analysis.json", "w") as f:
        json.dump(output, f)
