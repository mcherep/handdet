# Hand Detection in Tensorflow

A hand object detection using [ObjDet](https://github.com/mcherep/objdet). The object detector in the example uses SSD300 on top of MobileNetV2 applying transfer learning from a pretrained model on [MS COCO](http://cocodataset.org/#home). The model is trained with the [Egohands Dataset](http://vision.soic.indiana.edu/egohands_files/egohands_data.zip). The main goal is to demonstrate the easy use of `objdet` and to build a fast detector ready for realtime applications. A later application for tracking can be found in [ObjTracking](https://github.com/mcherep/objtracking)

## TODO

### Objdet

* The export doesn't work properly and breaks the imports
* Install, download, transfer_learning, save_model, load_model seem to work
* Train works with legacy but doesn't output results
* Tfrecords creation doesn't break but the records size seem suspiciously small.
The funtion that everyone uses tf.gfile doesn't work in 1.15 so use cv2 or others instead.
* Tfpredict doesn't work because the result of the prediction are now tensors. I wonder why
it worked at some point? Maybe I was using Tensorflow 2.0?

### Handdet

* Create a notebook that works out of the box with GPU
* Check transfer_learning_mobilenet in omo because it worked
loading the model and running predictions with tensorflow 1.14
* Make tensorboard work
* Output while training
* If legacy train works, I can remove the fork and go back to original

## Installation

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Download the [Egohands Dataset](http://vision.soic.indiana.edu/egohands_files/egohands_data.zip) in the `data` folder and execute the following:

```bash
unzip egohands_data.zip -d egohands && rm egohands_data.zip
```

## Google Colab

In order to run the notebook in [Google Colab](https://colab.research.google.com/github/)
