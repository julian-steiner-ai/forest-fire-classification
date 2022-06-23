import os
import tensorflow as tf

root_dir = '/workspaces/cv-project-forest-fire-detection/data/classification'
dirs = ['fire', 'smoke', 'non-smoke-fire']

for dir in dirs:
    for filename in os.listdir(os.path.join(root_dir, dir)):
        fullname = os.path.join(root_dir, dir, filename)
        with open(fullname, mode='rb') as f:
            image = f.read()
            try:
                tf.image.decode_image(image)
            except Exception:
                os.remove(fullname)