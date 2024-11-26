# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics.data import build_yolo_dataset2
from .train import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel


class NewDetectionTrainer(DetectionTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset2(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, 
                                   log_dir=self.save_dir)

    