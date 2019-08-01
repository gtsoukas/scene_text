# Scene Text Detection and Recognition

This project packages scene text algorithms for easy usage. Scenes as in photos are much harder for text detection and recognition than doing the same for scanned documents. The latter is mostly referred to as OCR and it is a well solved problem. Note that due to the usage of Deep Learning algorithms, text detection and recognition are rather slow, in particular on CPU-only machines and for images with a large number of words. Accuracies have reached an impressive level but are still below human performance. Currently, the following algorithms are available.

[EAST](https://arxiv.org/abs/1704.03155) is used for detection, where the implementation is mostly copied from [Jan Zdenek](https://github.com/kurapan/EAST ).

[MORAN](https://arxiv.org/abs/1901.03003) is used for recognition, where the implementation is mostly copied from [Canjie Luo](https://github.com/Canjie-Luo/MORAN_v2).


Installation:
```
pip install scene_text
```

Basic usage from Python:
```
import cv2

# The first import after installation can take a little longer as it downloads
# pre-trained models and compiles some native code.
from scene_text import AllWordsRecognizer

# initialize models
pipeline = AllWordsRecognizer()

# read an image
img = cv2.imread('path/to/my/image/file')[:, :, ::-1]

# detect and recognize all words in the image
words, boxes = pipeline.get_all_words(img)
```

Basic usage from command line:
```
scene_text my/input/image/folder my/output/folder
```

If you have trouble with the complex dependencies try Docker:
```
docker build -t scene_text .

docker run --rm -i -t -v ${PWD}:/scene_text scene_text

# ...
```
