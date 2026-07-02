import sys
sys.path.append('Tensorflow/models/research')
sys.path.append('Tensorflow/models/research/slim')

import os
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util




# 🔊 TTS Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

last_spoken = ""

def speak(text):
    global last_spoken
    if text != last_spoken:
        engine.say(text)
        engine.runAndWait()
        last_spoken = text

def speak_async(text):
    threading.Thread(target=speak, args=(text,)).start()

WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
MODEL_PATH = WORKSPACE_PATH + '/models'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME

labels = [{'name':'Hello', 'id':1}, 
          {'name':'Yes', 'id':2},
          {'name':'No', 'id':3},
          {'name':'Thank You', 'id':4},]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(
    ANNOTATION_PATH + '/label_map.pbtxt'
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Camera started...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        continue

    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32
    )

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))

    detections = {
        key: value[0, :num_detections].numpy()
        for key, value in detections.items()
    }

    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    if num_detections > 0:
        score = detections['detection_scores'][0]

        if score > 0.7:  # confidence threshold
            class_id = detections['detection_classes'][0] + 1
            label = category_index[class_id]['name']

            speak_async(label)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + 1,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=0.72,
        agnostic_mode=False
    )

    cv2.imshow('Object Detection', image_np_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()