from kafka import KafkaConsumer, KafkaProducer, errors
import json
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage


class MulticamTracker(object):

    def __init__(self,):
        super().__init__()
        self.consumer = KafkaConsumer(
            "metromind-raw", bootstrap_servers="kafka:9092", value_deserializer=lambda m: json.loads(m)
        )
        self.producer = KafkaProducer(
            bootstrap_servers="kafka:9092", value_serializer=lambda m: json.dumps(m).encode("utf-8")
        )
        self.overlapping_cameras = [(),]
        self.max_distance = 10

    def __remove_old_records(self, json_list):
        new_json_list = {}
        for json_ele in json_list:
            key = (json_ele["sensor"]["id"], json_ele["object"]["id"])
            value = new_json_list.get(key, None)
            if (value is None) or (json_ele.get("@timestamp", None) > value.get("@timestamp", None)):
                new_json_list[key] = value
        return list(new_json_list.values())

    def __get_distances(self, json_list1, json_list2):
        dist_matrix = np.zeros((len(json_list1), len(json_list2)), dtype=np.float32)
        for i1, json_ele1 in enumerate(json_list1):
            x1 = json_ele1["object"]["coordinate"]["x"]
            y1 = json_ele1["object"]["coordinate"]["y"]
            for i2, json_ele2 in enumerate(json_list2):
                x2 = json_ele2["object"]["coordinate"]["x"]
                y2 = json_ele2["object"]["coordinate"]["y"]
                dist_matrix[i1, i2] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # TODO overlapping cameras
        return dist_matrix

    def run(self,):
        while True:
            raw_messages = self.consumer.poll(timeout_ms=0.5 * 1000.0, max_records=5000)
            json_list = [msg.value for _, msg_list in raw_messages.items() for msg in msg_list]
            # TODO assert timestamp are nearby
            json_list = self.__remove_old_records(json_list)
            dist_matrix = self.__get_distances(json_list, json_list)
            # Agglomerative hierarchical clustering
            dist_array = squareform(dist_matrix)
            z_val = linkage(dist_array, "complete")
            clusters = fcluster(z_val, self.max_distance, criterion="distance")
            for i, (cluster_id, json_ele) in enumerate(zip(clusters, json_list), start=1):
                if i == cluster_id:
                    self.producer.send("metromind-start", json_ele)


if __name__ == "__main__":
    tracker = MulticamTracker()
    tracker.run()
