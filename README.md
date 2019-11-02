# Hand Detection in Tensorflow

A hand object detection using [ObjDet](https://github.com/mcherep/objdet). The object detector in the example uses SSD300 on top of MobileNetV2 applying transfer learning from a pretrained model on [MS COCO](http://cocodataset.org/#home). The model is trained with the [Egohands Dataset](http://vision.soic.indiana.edu/egohands_files/egohands_data.zip). The main goal is to demonstrate the easy use of `objdet` and to build a fast detector ready for realtime applications. A later application for tracking can be found in [ObjTracking](https://github.com/mcherep/objtracking) (**Coming soon**).

**Feel free to help me improve this project!**

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
