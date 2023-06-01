from pathlib import Path
import json
import numpy as np
import cv2
import time
from datetime import datetime
from ai.predictor import Predictor
# from kafka import KafkaProducer


def transform_image(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255
    return image


def main(onnx_file,):
    with open("cfg/cameras.json", "r") as f:
        cameras = json.load(f)
    num_cameras = len(cameras)
    caps = [cv2.VideoCapture(camera["camera_url"]) for camera in cameras]
    image_size = (416, 416)
    predictor = Predictor(onnx_file, image_size)
    # producer = KafkaProducer(bootstrap_servers="localhost:1234")
    # allow the camera to warmup
    time.sleep(0.1)
    # TODO LED green
    try:
        states_tracks = [[] for _ in range(num_cameras)]
        states_count = [0 for _ in range(num_cameras)]
        while True:
            frames = []
            for cap in caps:
                has_frame, frame = cap.read()
                # assert has_frame
                frames.append(frame)
            images = [transform_image(frame, image_size) for frame in frames]
            images = np.stack(images, axis=0)
            t = time.time()
            states_tracks, states_count = predictor.predict(
                t, states_tracks, states_count, images
            )
            crossing_persons = [
                {"time": datetime.fromtimestamp(t), "cameraId": camera["camera_id"],
                "personId": track.id, "direction": track.direction}
                for camera, state_tracks in zip(cameras, states_tracks)
                for track in state_tracks if track.direction != 0
            ]
            # if len(crossing_persons) > 0:
            #     producer.send(...)
    finally:
        cap.release()



if __name__ == "__main__":
    onnx_file = Path("onnx/fairmot_dla34.onnx")
    main(onnx_file)
