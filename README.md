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

In this repo there is an `ai.js` file which can be added as a Reference File and loaded dynamically to access some helper functions. This will help during rapid prototyping to share some code while models are developers and we learn how to generalize and simplify model input handling and output post-processing.

## Training a custom model

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