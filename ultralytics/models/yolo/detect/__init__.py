# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator
from .newTrain import NewDetectionTrainer
from .newValid import NewDetectionValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "NewDetectionTrainer", "NewDetectionValidator"
