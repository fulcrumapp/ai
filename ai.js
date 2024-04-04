function setRawValue(dataName, value) {
  CONFIG().results.push({
    type: 'set-value',
    key: FIELD(dataName).key,
    value: JSON.stringify(value)
  });
}

function setCaption(dataName, id, caption) {
  if (FIELDTYPE(dataName) !== 'PhotoField') {
    return;
  }

  const photos = VALUE(dataName) ?? [];

  const photo = photos.find(item => item.photo_id === id);

  if (photo) {
    photo.caption = caption;
  }

  setRawValue(dataName, photos);
}

function setupPhotoFieldTextRecognitionCaptions(dataName) {
  ON('add-photo', dataName, (event) => {
    RECOGNIZETEXT({ photo_id: event.value.id }, (error, result) => {
      if (result) {
        setCaption(dataName, event.value.id, result.text);
      }
    });
  });
}

function runInference(
  {
    photo_id,
    form_id,
    form_name,
    model,
    size,
    format,
    type,
    mean,
    std
  }, 
  callback
) {
  INFERENCE({
    photo_id: photo_id,
    form_id: form_id == null && form_name == null ? FORM().id : null,
    form_name: form_name,
    model,
    size,
    format: format ?? 'hwc',
    type: type ?? 'float',
    mean,
    std,
  }, (error, result) => {
    if (error) {
      ALERT(error.message);
      return;
    }

    for (const output of Object.values(result.outputs)) {
      if (Array.isArray(output.value)) {
        output.value = FLATTEN(output.value);
      }
    }

    if (callback) {
      callback(error, result);
    }
  });
}

const YOLOv5 = {};

YOLOv5.runClassification = function(
  {
    photo_id,
    form_id,
    form_name,
    model,
    size,
    format,
    type,
    mean,
    std,
    labels,
    threshold,
    top
  },
  callback
) {
  runInference({
    photo_id,
    form_id,
    form_name,
    model: model ?? 'yolov5m-cls.ort',
    size: size ?? 224,
    format: format ?? 'chw',
    type: type ?? 'float',
    mean: mean === null ? null : [0.485, 0.456, 0.406],
    std: std === null ? null : [0.229, 0.224, 0.225]
  }, (error, results) => {
    if (error) {
      callback(error);
      return;
    }

    const processed = YOLOv5.processClassificationOutput({
      values: Object.values(results.outputs)[0].value,
      labels,
      threshold,
      top
    });

    callback(null, processed);
  });
}

YOLOv5.processObjectDetectionOutput = function({
  detectionBoxes,
  detectionClasses,
  detectionScores,
  detectionCount,
  labels,
  threshold
}) {
  const resultsArray = [];
  const count = detectionCount.length > 0 ? detectionCount[0] : 0;

  if (count === 0) {
    return resultsArray;
  }

  for (let i = 0; i < count; ++i) {
    const score = detectionScores[i];

    // Filter results with score < threshold.
    if (score < threshold) {
      continue;
    }

    const classIndex = detectionClasses[i];
    const className = labels[classIndex];

    const x1 = detectionBoxes[4 * i];
    const y1 = detectionBoxes[4 * i + 1];
    const x2 = detectionBoxes[4 * i + 2];
    const y2 = detectionBoxes[4 * i + 3];

    const box = {
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1
    };

    resultsArray.push({
      className,
      classIndex,
      score,
      box
    });
  }

  return resultsArray.sort((first, second) => {
    return first.score - second.score;
  });
}

YOLOv5.processClassificationOutput = function({
  values,
  labels,
  threshold,
  top
}) {
  const results = values.map((score, index) => {
    return {
      index,
      score,
      label: labels[index]
    };
  });

  const sorted = results.sort((a, b) => a.score - b.score).reverse();

  const topK = top != null ? sorted.slice(0, top) : sorted;

  return topK.filter(result => result.score > (threshold ?? 0));
}

YOLOv5.runObjectDetection = function(
  {
    photo_id,
    form_id,
    form_name,
    model,
    size,
    format,
    type,
    labels,
    threshold
  },
  callback
) {
  runInference({
    photo_id,
    form_id,
    form_name,
    model: model ?? 'yolov5m.ort',
    size: size ?? 640,
    format: format ?? 'hwc',
    type: type ?? 'float'
  }, (error, { outputs }) => {
    if (error) {
      callback(error);
      return;
    }

    const detections = YOLOv5.processObjectDetectionOutput({
      detectionBoxes: outputs.output_0.value,
      detectionScores: outputs.output_1.value,
      detectionClasses: outputs.output_2.value,
      detectionCount: outputs.output_3.value,
      labels: labels,
      threshold: threshold ?? 0.3
    });

    callback(null, detections);
  });
}

function chatGPT({ prompt, apiKey, model, temperature }, callback) {
  const options = {
    method: 'POST',
    url: 'https://api.openai.com/v1/chat/completions',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
     "model": model ?? 'gpt-3.5-turbo',
     "messages": [{ role: "user", content: prompt }],
     "temperature": temperature ?? 0.7
    })
  };

  REQUEST(options, (req, res, body) => {
    const json = JSON.parse(body);

    callback(json.choices && json.choices.length ? json.choices[0].message.content : null, json);
  });
}

module.exports = {
  setCaption,
  setRawValue,
  setupPhotoFieldTextRecognitionCaptions,
  runInference,
  YOLOv5,
  chatGPT
};
