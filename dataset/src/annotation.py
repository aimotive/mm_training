import json

from typing import Dict


class Annotation:
    """
    This data structure stores the annotated objects for a given keyframe.

    Attributes:
        objects: list of annotated objects, each object is represented as a dict, example object:
                {
                    "ActorName": "CAR 04",
                    "BoundingBox3D Extent X": 4.7388153076171875,
                    "BoundingBox3D Extent Y": 2.0734505653381348,
                    "BoundingBox3D Extent Z": 1.9572224617004395,
                    "BoundingBox3D Orientation Quat W": 1,
                    "BoundingBox3D Orientation Quat X": 0,
                    "BoundingBox3D Orientation Quat Y": 0,
                    "BoundingBox3D Orientation Quat Z": 0,
                    "BoundingBox3D Origin X": 126.27908325195312,
                    "BoundingBox3D Origin Y": -1.1651087999343872,
                    "BoundingBox3D Origin Z": 0.36108487844467163,
                    "ObjectId": 4,
                    "ObjectType": "CAR",
                    "Occluded": 0,
                    "Relative Velocity X": -0.6558380126953125,
                    "Relative Velocity Y": -1.8364770412445068,
                    "Relative Velocity Z": 0.103598952293396,
                    "Truncated": 0
                }
        timestamp: timestamp when the annotation is created
    """
    def __init__(self, path: str):
        """
        Args:
            path: path to the annotation json file
        """
        self.path = path
        with open(path, 'r') as gt_file:
            annotations = json.load(gt_file)

        self.objects = [self.filter_attributes(obj) for obj in annotations['CapturedObjects']]
        # self.timestamp = annotations['Timestamp']

    def filter_attributes(self, annotation: Dict) -> Dict:
        """
        Filter out occluded and truncated attributes since dummy values are stored.

        Args:
            annotation: annotated objects represented as a dict
        Returns:
            annotation: filtered annotated objects
        """
        if 'Occluded' in annotation:
            del annotation['Occluded']
        if 'Truncated' in annotation:
            del annotation['Truncated']

        return annotation
