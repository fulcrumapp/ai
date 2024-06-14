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

function setupPhotoFieldTextRecognitionCaptions(dataName, process) {
  ON('add-photo', dataName, (event) => {
    RECOGNIZETEXT({ photo_id: event.value.id }, (error, result) => {
      if (result) {
        const text = process ? process(result.text) : result.text;

        setCaption(dataName, event.value.id, text);
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

YOLOv5.classify = function(
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

    const predictions = YOLOv5.processClassificationOutput({
      values: Object.values(results.outputs)[0].value,
      labels,
      threshold,
      top
    });

    callback(null, { predictions, raw: results });
  });
}

YOLOv5.processDetectionOutput = function({
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

YOLOv5.detect = function(
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

    const predictions = YOLOv5.processDetectionOutput({
      detectionBoxes: outputs.output_0.value,
      detectionScores: outputs.output_1.value,
      detectionClasses: outputs.output_2.value,
      detectionCount: outputs.output_3.value,
      labels: labels,
      threshold: threshold ?? 0.3
    });

    callback(null, { predictions, raw: outputs });
  });
}

const YOLOv8 = {};

YOLOv8.classify = YOLOv5.classify;
YOLOv8.processClassificationOutput = YOLOv5.processClassificationOutput;

function chatGPT({ prompt, apiKey, model, temperature, ...options }, callback) {
  const requestOptions = {
    method: 'POST',
    url: 'https://api.openai.com/v1/chat/completions',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: model ?? 'gpt-4o',
      messages: [{ role: "user", content: prompt }],
      temperature: temperature ?? 0.0,
      ...options
    })
  };

  REQUEST(requestOptions, (err, res, body) => {
    if (err) {
      callback(err);
    } else {
      const json = JSON.parse(body);

      callback(null, json.choices && json.choices.length ? json.choices[0].message.content : null, json);
    }
  });
}

function computeIoU(boxA, boxB) {
  const xA = Math.max(boxA.x1, boxB.x1);
  const yA = Math.max(boxA.y1, boxB.y1);
  const xB = Math.min(boxA.x2, boxB.x2);
  const yB = Math.min(boxA.y2, boxB.y2);

  const interArea = Math.max(0, xB - xA + 1) * Math.max(0, yB - yA + 1);

  const boxAArea = (boxA.x2 - boxA.x1 + 1) * (boxA.y2 - boxA.y1 + 1);
  const boxBArea = (boxB.x2 - boxB.x1 + 1) * (boxB.y2 - boxB.y1 + 1);

  const iou = interArea / (boxAArea + boxBArea - interArea);

  return iou;
}

function nonMaximumSuppression(boxes, scores, iouThreshold) {
  const nmsBoxes = [];
  const boxesWithScores = boxes.map((box, index) => ({...box, score: scores[index]}));
  boxesWithScores.sort((a, b) => b.score - a.score);

  while (boxesWithScores.length > 0) {
    const [currentBox] = boxesWithScores.splice(0, 1);
    nmsBoxes.push(currentBox);

    for (let i = boxesWithScores.length - 1; i >= 0; i--) {
      const iou = computeIoU(currentBox, boxesWithScores[i]);
      if (iou > iouThreshold) {
        boxesWithScores.splice(i, 1);
      }
    }
  }

  return nmsBoxes;
}

module.exports = {
  nonMaximumSuppression,
  setCaption,
  setRawValue,
  setupPhotoFieldTextRecognitionCaptions,
  runInference,
  YOLOv5,
  YOLOv8,
  chatGPT
};
