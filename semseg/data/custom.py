import os

from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from .graspnet import register_graspnet_instances
from .graspnet_meta import get_graspnet_instances_meta
from .os_coco import register_os_coco_instances

_GRASPNET_OS_SPLITS = {
    "graspnet_train": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_train.json"),
    "graspnet_test_1": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_1.json"),
    "graspnet_test_2": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_2.json"),
    "graspnet_test_3": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_3.json"),
    "graspnet_test_4": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_4.json"),
    "graspnet_test_5": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_5.json"),
    "graspnet_test_6": ("graspnet_os/images", "graspnet_os/annotations/graspnet_os_test_6.json"),
}

def register_graspnet_os(root):
    for key, (image_root, json_file) in _GRASPNET_OS_SPLITS.items():
        register_graspnet_instances(
            key,
            get_graspnet_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_COCO_OS_SPLITS = {
    "coco_2017_train_invoc": ("coco/train2017", "coco/annotations/instances_train2017.json"),
}

COCO_INVOC_CATEGORIES = [
    'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
    'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
    'train', 'tv'
]
coco_name_id_dic = {cats["name"]:cats["id"] for cats in COCO_CATEGORIES}
COCO_INVOC_IDS = [coco_name_id_dic[name_cat] for name_cat in COCO_INVOC_CATEGORIES]

def register_coco_os(root):
    for key, (image_root, json_file) in _COCO_OS_SPLITS.items():
        register_os_coco_instances(
            key,
            _get_coco_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if __name__.endswith(".custom"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_graspnet_os(_root)
    register_coco_os(_root)