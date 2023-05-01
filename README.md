# FMRACV

Machine learning classifier to categorize images.

It was written by @jpetazzo with two goals:

1. Learn and implement machine learning concepts.
2. Help https://ephemerasearch.com/ classify old postcards.

You can probably use that code to classify your own images if you want,
but it's probably not a great idea, because this is literally my first
machine learning model, and there are honestly many parts that I don't
fully understand yet. There are certainly better options out there.

## In short

It's in Python, and uses Tensorflow and Keras.

I took the code of the model from one of the many machine learning
tutorials that I've read in the last few months. I inititally had
no idea about how it worked and why, but it worked suprisingly well
(achieving 95% accuracy on my first naive tests).

Since then, I've been watching this [excellent computer vision
course by Justin Johnson at University of Michigan][course],
and if I understand correctly, the model that I use turns out to
be a relatively basic CNN, similar to the famous AlexNet.

(Don't take my word for it, though; at the end of 2022, I knew
nothing about computer vision, and now in 2023, the main thing
I've learned is how vast is the extend of what I don't know, and
even that, I'm not sure 😅)

## Running it

First, you need a bunch of images. Probably at least a few hundreds or
thousands. Put all the images in the `imgroot` subdirectory.
The exact naming and layout (subdirectories etc.) of the images
doesn't matter.

💡 For our first tests, we had a few thousands of images per label.
For our first production deployment, we had between 10000 and
25000 images depending on the model.

Then, your images need to be labeled. You do this by creating lists of images.
A list of image is just a text file, with one file name per line. File names
are relative to the `imgroot` directory. You can have multiple lists for each
category.

💡 One of our models tries to predict whether a postcard is blank,
or has an address on it. We created multiple lists for each category;
see [with-address-or-blank.yaml](with-address-or-blank.yaml) for an example.

Then you need to create a configuration file for the model. See
[with-address-or-blank.yaml](with-address-or-blank.yaml) for an example.
Note that the `port` is optional (it is used only when serving the model
over an API endpoint). `training_data:` is a list (and not a mapping) to
make sure that the ordering of the labels is preserved.

The dependencies are listed in `requirements.txt`. They should install easily
with `pip`.

Then, I recommend that you run `preprocess-image-list.py` on each list of images.
It will load each image, transform it into a tensor (the vector format used by
the model), and save it back to disk. This is not strictly necessary (the code
will load and transform the images on the fly when needed), but it speeds up
the loading of the images by 100x.

⚠️ If you make changes to the input format of the model (number of channels or
resolution), you will need to delete the cached tensors.

⚠️ The code currently rejects images that don't have an aspect ratio close
to the one of a postcard in landscape orientation. You probably want to remove
that test from the code if you run it on your own images!

You can then train the model by running `fmracv.py train model-config.yaml`.

This will work on CPU, but it will be WAY faster on GPU! (Like, easily 100x faster).
The code uses Tensorflow, and I couldn't get Tensorflow+GPU to work (the installation
instructions suggest to mess with the CUDA libraries installed on the system, to
which I replied "No, thanks"). If you want to run on GPU without taking the risk
of screwing up your system, you can install Docker, the NVIDIA runtime, and then
use the provided script `docker-run-tensorflow.sh`. That script starts a container
using some official Tensorflow image. It bind-mounts a few directories, so if you
want extra dependencies, you can `pip install` them and they will persist across
multiple runs of the container.

💡 I don't think that you need a bleeding edge GPU for this. I trained my models
on a GeForce 1660 with 6 GB VRAM, and it looks like the training process used
a bit more than 1 GB VRAM. If you end up running out of VRAM, it *might* be
possible to use less VRAM by reducing the `BATCH_SIZE` in the code. But see
notes below.

After training the model, it will be saved to a `.h5` file.

You can check how the model performs on your dataset by running
`fmracv.py analyze model-config.yaml`. This will test the model on every
single image of the training set, and compare the prediction with the correct
label. Predictions with a confidence value of less than 0.95 (and invalid
predictions) are saved in a JSON file. You can then have a look at that
JSON file. There is a Jupyter notebook in the `notebooks` directory
that contains some code leveraging Bokeh to produce a "nice" representation
of the data, so that you can see very easily the images that "confused" the
model.

⚠️ It looks like even a few badly labeled images can really affect the performance
of the model, so make sure that your images are labeled correctly (i.e. are
in the right lists).

💡 While inference runs faster on GPU, I found it perfectly reasonable
even on CPU.

If you want to run the model on a large number of images, put them in a list
file (just like the training lists) and run `fmracv.py predict model-config.yaml list-file`.
It will create multiple output files:

- `list-file.XXX` where `XXX` is one of the labels of the model, for predictions
  with a confidence above 0.95
- `list-file.inconclusive` when the confidence was below 0.95
- one JSON file with all the predictions
- another JSON file with the inconclusive predictions

You can then use the Jupyter notebook described above to have a look at the
inconclusive predictions, for instance.

You can also serve the model over an API endpoint by running `fmracv.py serve model-config.yaml`.
Then test it with:

```shell
curl -F "image=@path/to/image.jpg" localhost:8001/predict
```

(Where `8001` should be the port number in the `model-config.yaml` file.)

⚠️ The serving is currently done with the default Flask server, which is
considered not "production-ready". My current approach is that it's fine for my needs
(ad-hoc invocation of the model). Note that if you need to process large
batches, using the `predict` CLI command will be MUCH faster (100x maybe?)
because it will run inference in batches.

## Helper scripts

A few scripts that you might find helpful...

`view-list.sh`: this will use `feh` to view all the images in a list,
In thumbnail form. I use it to show 35 images at a time on a 4K screen.
Hit `q` to quit `feh`, then `ENTER` to load the next batch.

`weed-out.sh`: this is when you have a list with "mostly" images of
a certain class, but with a few rare outliers. For instance, after running
prediction on a new batch of images; or if you messed up labelling earlier
and want to fix it. This script will use `feh` to load a list of images,
showing 35 images at a time on the screen. If you click on an image, it will put
it in the `.err` output list. When you quit `feh` (with `q`) it will
move all the other images (that you haven't clicked) to the `.ok` output
list. Then press `ENTER` to move to the next batch of 35 images.

`quick-label.sh`: and this one is when you have a bunch of images with
different labels, and you want to quickly sort them out. Check the source
for details.

`deploy.sh` and `run.sh`: probably the ugliest deployment pipeline you'll see today.
When/if this needs to handle more capacity, I'll add a Dockerfile and maybe
deploy to Cloud Run or whatever.

## Limitations and TODO

- [X] Make it easy to train and run multiple labels
      (implemented with the YAML configuration)
- [ ] When serving the model with the API endpoint, allow the user to
      specify an URL instead of having to POST the image
- [ ] Store the performance of each model with tensorboard for easier comparison
- [ ] Compare performance of the model with larger input layers
      (I wonder if the current resolution of 224x224 might be too low to pick up
      very faint features?)
- [ ] Check performance when running the model on e.g. a "left crop" and "right crop"
      of the image (currently we do a resize+pad, so a large portion of the input
      field isn't used)
- [ ] Check performance with other architectures (I tried a bunch of variations
      already, but that was ad-hoc; I need to re-run formal tests with tensorboard)
- [ ] Try with a different learning rate schedule (i.e. lower the learning rate
      when the loss stagnates, which it seems to do after just 2-3 epochs anyway)
- [ ] Check if data augmentation helps (I made the assumption that it wouldn't be
      a game changer since all postcards are already in a very specific format and
      orientation, but I need to validate it)
- [ ] Figure out if it would be beneficial to switch to tf.data.Dataset (and how).
      It looks like it should be "the right thing to do", and it looks like it
      would have the potential to e.g. automatically convert the JPEG images to
      tensors in parallel and cache the results, but I found the documentation
      extremely unclear, and my experiments gave me totally obscure tracebacks.

## Notes

### VRAM usage

My initial implementation would load all the images, then build a stacked tensor
with all the images. It worked, but of course, when I started working on a bigger
training set, I ran out of VRAM. I tried to see if there was some way to build
the stacked tensor efficiently (e.g. by having the stacked tensor reference the
individual tensor images?) but it looks like this is not possible (I'm not sure
though).

I thought that loading the images to VRAM at each epoch would be bad for performance,
so I decided to try the following approach:

1. Compute exactly how much VRAM is available for my training set.
2. Compute how many images would fit in VRAM.
3. Crop the training set (reduce the number of images) so that it fits in that available VRAM.
4. Load image tensors in CPU memory.
5. Build the giant tensor in CPU memory.
6. Then move it to GPU memory and train the model.

It turns out that after implementing all these steps, I realized that I didn't
need to crop the training set anymore. For some reason, at training time,
Tensorflow is able to pull only bits of the training set into GPU VRAM
(probably one mini-batch at a time).

"Interestingly", I haven't been able to figure out how much VRAM was available
to Tensorflow from Tensorflow itself. I can see it with tools like `nvidia-smi`,
and when Tensorflow initializes the GPU, it shows the GPU VRAM size; so it
definitely has that information - but I couldn't find how to access it.

[course]: https://www.youtube.com/watch?v=dJYGatp4SvA