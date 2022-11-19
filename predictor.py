from typing import List
from sieve.types import FrameSingleObject, BoundingBox, FrameFetcher, Object
from sieve.predictors import ObjectPredictor
from sieve.types.constants import FRAME_NUMBER, BOUNDING_BOX, SCORE, CLASS, START_FRAME, END_FRAME, OBJECT
from sieve.types.outputs import StaticClassification
from deepface import DeepFace
import cv2

class FaceAttributePredictor(ObjectPredictor):

    def setup(self):
        pass

    def predict(self, frame_fetcher: FrameFetcher, object: Object) -> StaticClassification:
        if object.cls != 'face':
            return {}
        # Get the frame array
        object_start_frame, object_end_frame = object.get_static_attribute(START_FRAME), object.get_static_attribute(END_FRAME)
        frame_number = (object_start_frame + object_end_frame)//2
        object_bbox: BoundingBox = object.get_temporal_attribute(BOUNDING_BOX, frame_number)
        # Get image from middle frame
        frame_data = frame_fetcher.get_frame(frame_number)
        # Initialize classification with the object being classified
        out_dict = {OBJECT: object}
        # Crop frame data to bounding box constraints
        frame_data = frame_data[int(object_bbox.y1):int(object_bbox.y2), int(object_bbox.x1):int(object_bbox.x2)]
        #Check if bounding box is invalid
        if frame_data.shape[0] == 0 or frame_data.shape[1] == 0:
            out_dict["age"] = "unknown"
            out_dict["gender"] = "unknown"
            out_dict["race"] = "unknown"
            out_dict["emotion"] = "unknown"
            return StaticClassification(**out_dict)
        # Run the model
        obj = DeepFace.analyze(frame_data, 
            actions = ['age', 'gender', 'race', 'emotion'], enforce_detection = False
        )
        if not ("age" in obj):
            out_dict["age"] = "unknown"
            out_dict["gender"] = "unknown"
            out_dict["race"] = "unknown"
            out_dict["emotion"] = "unknown"
            return StaticClassification(**out_dict)
        else:
            out_dict["race"] = obj["dominant_race"]
            out_dict["gender"] = obj["gender"].lower()
            out_dict["age"] = obj["age"]
            out_dict["emotion"] = obj["dominant_emotion"]
        return StaticClassification(**out_dict)
