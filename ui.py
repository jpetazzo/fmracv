import db
import flask
import json
import logging


PAGE_SIZE = 48


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
    html = ""
    with db.Session() as session:

        images = session.query(db.Image).count()
        images_with_sha256 = session.query(db.Image).filter(db.Image.sha256 != "").count()
        images_with_wh = session.query(db.Image).filter(db.Image.width > 0).count()
        html += f"<p>{images} images</p>"
        html += f"<p>{images_with_sha256} images with SHA256</p>"
        html += f"<p>{images_with_wh} images with dimensions</p>"

        html += "<table>\n"
        for model in session.query(db.Model):
            html += "<tr>\n"
            html += f'<td><a href="/model/{model.name}">{model.name}</a></td>\n'
            html += f"<td>{model.type}</td>\n"
            html += f"<td>{model.python_class}</td>\n"
            html += f"<td>{model.labels}</td>\n"
        html += "</table>\n"

    return html


@app.route("/model/<name>")
def model(name):
    html = ""
    html += '<a href="/">Home</a><br/>\n'
    with db.Session() as session:
        model = session.query(db.Model).filter_by(name=name).one()
        labels = json.loads(model.labels)
        for label in labels:
            n = session.query(db.Label).filter(db.Label.model_name==name, db.Label.label==label).count()
            html += f'<a href="/model/{name}/trainingdata/{label}/1">Training data for {label}</a> ({n})<br/>\n'
        n = session.query(db.Label).filter(db.Label.model_name==name, db.Label.label==None, db.Label.prediction!=None).count()
        for label in labels:
            html += f'<a href="/model/{name}/predictions/{label}">Predictions for {label}</a> (&lt;{n})<br/>\n'
        html += f'<a href="/label/{name}/next">Manual labeling interface</a>\n'
    return html


@app.route("/model/<model_name>/trainingdata/<label>/<int:page>")
def review_trainingdata(model_name, label, page=1):
    with db.Session() as session:
        q = (
            session.query(db.Image, db.Label)
            .join(db.Label)
            .filter(db.Label.model_name==model_name)
            .filter(db.Label.label==label)
        )
        count = q.count()
        pages = (count + PAGE_SIZE - 1) // PAGE_SIZE
        results = {}
        for r in q.offset(PAGE_SIZE*(page-1)).limit(PAGE_SIZE):
            results[r[0].sha256] = ( r[0], r[1], "")
    return flask.render_template(
        "base.html",
        model=model_name, label=label, count=count,
        page=page, pages=pages,
        data=results.values()
    )


@app.route("/model/<model_name>/predictions/<label>")
def review_predictions(model_name, label):
    confidence_threshold = float(flask.request.args.get("confidence_threshold", 0.95))
    with db.Session() as session:
        q = (
            session.query(db.Image, db.Label)
            .join(db.Label)
            .filter(db.Label.model_name==model_name)
            .filter(db.Label.prediction!="", db.Label.label==None)
        )
        count = q.count()
        results = {}
        for r in q:
            if len(results) >= PAGE_SIZE:
                break
            prediction = json.loads(r[1].prediction)
            if prediction[label] > confidence_threshold:
                hover_text = " ".join(f"{k}={v}" for (k,v) in prediction.items())
                results[r[0].sha256] = ( r[0], r[1], hover_text )
            else:
                count -= 1
    return flask.render_template(
        "base.html",
        model=model_name, label=label, count=count,
        data=results.values()
    )


@app.route("/label/<model_name>/<sha256>", methods=["GET"])
def label_ui(model_name, sha256):
    with db.Session() as session:
        model = session.query(db.Model).filter_by(name=model_name).one()
        labels = json.loads(model.labels)
        if sha256 == "next":
            todo = session.query(db.Label).filter_by(model_name=model_name, label=None)
            thing_to_label = todo.first()
            if not thing_to_label:
                return flask.redirect(f"/model/{model_name}")
            count = todo.count()
        elif sha256 == "hint":
            confidence_threshold = float(flask.request.args.get("confidence_threshold", 0.6))
            todo = session.query(db.Label).filter_by(model_name=model_name, label=None)
            count = todo.count()
            for thing_to_label in todo:
                count -= 1
                prediction = json.loads(thing_to_label.prediction)
                if all(p < confidence_threshold for p in prediction.values()):
                    break
        else:
            thing_to_label = session.query(db.Label).filter_by(model_name=model_name, sha256=sha256).one()
            count = 1
        prediction = json.loads(thing_to_label.prediction)
        image = session.query(db.Image).filter_by(sha256=thing_to_label.sha256).first()
    return flask.render_template(
        "base.html",
        model=model_name, labels=labels, todo=count, prediction=prediction,
        todo_image=image, todo_label=thing_to_label, label=thing_to_label.label,
    )


@app.route("/label/<model_name>", methods=["POST"])
def update_labels(model_name):
    payload = flask.request.get_json()
    with db.Session() as session:
        for sha256, label in payload.items():
            if label:
                existing_label = session.query(db.Label).filter_by(model_name=model_name, sha256=sha256).first()
                if existing_label:
                    existing_label.label = label
                else:
                    session.add(db.Label(model_name=model_name, sha256=sha256, label=label))
            else:
                session.query(db.Label).filter_by(model_name=model_name, sha256=sha256).delete()
        session.commit()
    return ""


@app.route("/image/<origin>/<path:path>")
def serve_image(origin, path):
    absolute_path = db.make_path(origin, path)
    return flask.send_file(absolute_path)
