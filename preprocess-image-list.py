#!/usr/bin/env python
import fmracv
import os
import pickle
import sys
import tqdm

"""
Takes an image list as argument.
Will try to load all images, generate a tensor, and save the resulting tensor
(to make it faster to load the image later).
Outputs two list files:
- a ".ok" list, with all the files that were processed successfully;
- a ".bad" list, with the ones that failed to load or had the wrong format.
"""

input_list = open(sys.argv[1])
output_ok_list = open(sys.argv[1] + ".ok", "w")
output_err_list = open(sys.argv[1] + ".err", "w")

for image_file in tqdm.tqdm(list(input_list)):
    print
    tensor = fmracv.load_image(image_file.strip())
    if tensor is None:
        output_err_list.write(image_file)
    else:
        pickle_file = os.path.join(fmracv.BASE_DIR, image_file.strip() + ".pck")
        with open(pickle_file, "wb") as f:
            pickle.dump(tensor, f)
        output_ok_list.write(image_file)
