# Fulcrum AI Resources

## Overview

Computer Vision in Fulcrum is exposed by the `INFERENCE()` function in Data Events. Using a combination of Fulcrum features, we can load and run inference on photos taken inside the app. Trained models are uploaded as Reference Files on an app. Fulcrum supports models in the [ONNX Runtime `.ort`](https://onnxruntime.ai/) format. It supports some flexibility with input tensor formats to accommodate models trained from the Tensorflow ecosystem or the PyTorch ecosystem.

Below is the basic process of preparing a model to use in Fulcrum:

* Train or find a model in either PyTorch .pt format or Tensorflow format
* Convert it to an ONNX model (`.onnx` file)
* Convert the `.onnx` file to an `.ort` file for use in the ONNX Runtime and Fulcrum
* Upload the `.ort` file as a reference file
* Upload the labels `.txt` file as a reference file
* Add a data events snippet to run the inference

## Example Models

* [Utility Poles](https://drive.google.com/file/d/1sISnmO4TRAqm4DBgLtKeaKiFcB2a75Kx/view?usp=sharing) / [labels](https://drive.google.com/file/d/1ADzWi5QLJLJbtrIyhCnr81CziNskfQW1/view?usp=drive_link)
* [YOLOv5 Object Detection](https://drive.google.com/file/d/1VZ7OJrRIivsFGYQaPBgb6O10qubKg3E2/view?usp=drive_link) / [labels](https://drive.google.com/file/d/1WfA-O2RTjogKZqi9WKwj1zvIkGY6o9ko/view?usp=drive_link)
* [YOLOv5 Classification](https://drive.google.com/file/d/1UO_rDDowGj5BnFqKooTkE1_Su5WGhilF/view?usp=drive_link) / [labels](https://drive.google.com/file/d/1OIDh6fX702tzHHf3mIb62nUnx5ZHfLDr/view?usp=sharing)

## Model Outputs

Processing model output is the most complex part of integrating a custom model. Each model architecture generally has its own output shape and processing requirements. And depending on how the model was created, it may or may not have embedded post processing like NMS (non max suppression) which is required for practical use of the model. For object detection models that require NMS, it must currently be done inside the model with the [`NonMaxSuppression`](https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression) operator at the end. Netron is a useful tool for inspecting the model inputs and outputs.

## Helpers

In this repo there is an `ai.js` file which can be added as a Reference File and loaded dynamically to access some helper functions. This will help during rapid prototyping to share some code while models are developing and we learn how to generalize and simplify model input handling and output post-processing.

## Training a custom model

### Setup python environment

```sh
python3 -m venv fulcrum-ai
source fulcrum-ai/bin/activate
pip install -r requirements.txt
```

### Train an image classification model

```sh
yolo train imgsz=640 epochs=300 batch=4 data=/path/to/photos model=yolov8s-cls.pt
```

### Export the model to ONNX

```sh
yolo export imgsz=640 model=runs/classify/train6/weights/best.pt format=onnx
python -m onnxruntime.tools.convert_onnx_models_to_ort runs/classify/train6/weights/best.onnx

# .ort model now at runs/classify/train6/weights/best.ort
```

## Training a custom object detection model

### Setup
```sh
git clone https://github.com/ultralytics/yolov5 yolo-repo
cd yolo-repo && git pull && cd -
pip install -r yolo-repo/requirements.txt
```

### Train the model
```sh
python yolo-repo/train.py --imgsz 640 --epochs 300 --batch-size 2 --data corrosion/data.yaml --weights yolov5m.pt
```

### Export the model

To export an object detection model, we want the Non Maximum Suppression (NMS) operation baked into the end of the model. This allows us to use the model output as-is without need NMS algorithms implemented in the Data Events post processing. The basic concept is the raw YOLO object detection models output many slightly overlapping bounding boxes for the "same object" and you need to do post processing to find the "best" bounding boxes based on the scores so you don't end up with many boxes for the same objects.

```
python yolo-repo/export.py --imgsz 640 --weights yolo-repo/runs/train/exp6/weights/best.pt --include saved_model --dynamic --nms
python -m tf2onnx.convert --opset 18 --saved-model yolo-repo/runs/train/exp6/weights/best_saved_model --output corrosion.onnx --tag serve
python -m onnxruntime.tools.convert_onnx_models_to_ort corrosion.onnx
```

* [Notebook for training a YOLOv5 object detector](https://colab.research.google.com/drive/1DlDVnYTftdAZ83SkUXTEXAO4utp2h0Eu?usp=sharing)

## Data Events Usage

```js
ON('add-photo', 'photos', (event) => {
  INFERENCE({
    photo_id: event.value.id,
    model: 'yolov5m-cls.ort',
    size: 224,
    format: 'chw',
    type: 'float',
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225]
  }, (error, { output }) => {
    if (error) {
      ALERT(error.message);
      return;
    }

    const results = Object.values(outputs)[0].value.map((score, index) => {
      return {
        index,
        score,
        label: IMAGENET[index]
      };
    });

    const sorted = results.sort((a, b) => b.score - a.score);

    const topK = top != null ? sorted.slice(0, top) : sorted;

    SETVALUE('detections', JSON.stringify(topK));
  });
});
```

## New Functions

```js
// validations.js added as reference file to another form
// function validateName(name) {
//   return name && name.length > 5;
// }
// module.exports = {
//   validateName
// };

// load validations.js from another form (also accepts form_id) and assign it a global variable `validations`
LOADFILE({ name: 'validations.js', form_name: 'Some Other Form Name', variable: 'validations' });

ON('validate-record', () => {
  if (!validations.validateName($name)) {
    INVALID('Name is not valid');
  }
});

// load records by their ids
LOADRECORDS({ ids: $record_link_field }, (error, records) => {
  ALERT(`Loaded ${records.length} records`);
});

// load all the records in a form (also accepts form_id) and filter them in JS, the count and size of records impact performance
LOADRECORDS({ form_name: 'Some Reference Form' }, (error, records) => {
  ALERT(`Loaded ${records.length} records`);
});

// load another form schema, also accepts form_id
LOADFORM({ form_name: 'Some Reference Form' }, (error, form) => {
  ALERT(`Loaded ${form.name} schema`);
});
```

### Datasets

https://github.com/InsulatorData/InsulatorDataSet
https://github.com/phd-benel/MPID
https://github.com/andreluizbvs/InsPLAD
https://github.com/andreluizbvs/PLAD