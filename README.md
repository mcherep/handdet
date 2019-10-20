# Hand Detection in Tensorflow

A hand object detection using [Object Detection in Tensorflow](https://github.com/mcherep/objdet). The object detector uses SSD300 on top of MobileNetV2 applying transfer learning from a pretrained model on [MS COCO](http://cocodataset.org/#home). The model is trained with the [Egohands Dataset](http://vision.soic.indiana.edu/egohands_files/egohands_data.zip). The main goal is to demonstrate the easy use of `objdet` and to build a fast detector ready for realtime applications.

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
