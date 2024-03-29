import os
import logging
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Boxes, Instances
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.utils.events import get_event_storage
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy

from semseg.data.graspnet_meta import GRASPNET_KNOWN_IDS
from semseg.data.custom import COCO_INVOC_IDS

logger = logging.getLogger(__name__)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def _log_classification_stats(pred_logits, gt_classes, prefix="softmax_classifier"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)

def fast_rcnn_inference_single_image_known(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    proposal_embedding: torch.Tensor,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        boxes (Tensor): Tensors of predicted class-specific or class-agnostic
            boxes for each image with shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (Tensor): Tensors of predicted class scores for each image with 
            shape (Ri, K + 1), where Ri is the number of predicted objectsfor image i. 
            Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (tuple[int, int]): (width, height) tuple for each image.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (Instances): Instance stores the topk most confidence detections.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    proposal_embedding = proposal_embedding[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, proposal_embedding = boxes[keep], scores[keep], filter_inds[keep], proposal_embedding[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.proposal_embedding = proposal_embedding
    return result

def fast_rcnn_inference_single_image_unknown(
    boxes,
    scores,
    proposal_embedding: torch.Tensor,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference_single_image_known`.

    Returns:
        Same as `fast_rcnn_inference_single_image_known`.
    """
    scores = scores.unsqueeze(dim=1)
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    proposal_embedding = proposal_embedding[filter_inds[:, 0]]

    # print(len(scores))

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds, proposal_embedding = boxes[keep], scores[keep], filter_inds[keep], proposal_embedding[keep]

    # print(len(scores))

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = (torch.zeros(len(scores), device='cuda') + 1000).long()
    result.proposal_embedding = proposal_embedding
    return result

class SoftMaxClassifier(nn.Module):
    """
    Softmax classifier with one linear layer to classify the known objects.
    """
    @configurable
    def __init__(
        self,
        num_classes: int,
        num_known_classes: int,
        unknown_id: int,
        dataset_name: str,
        input_size: int,
        known_score_thresh: float,
        known_nms_thresh: float,
        known_topk: int,
        unknown_score_thresh: float,
        unknown_nms_thresh: float,
        unknown_topk: int,
        cls_loss_weight: float
    ):
        """

        """
        super().__init__()
        self.num_classes = num_classes
        self.num_known_classes = num_known_classes
        self.unknown_id = unknown_id

        self.cls_score = nn.Linear(input_size, num_known_classes + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        self.known_score_thresh = known_score_thresh
        self.known_nms_thresh = known_nms_thresh
        self.known_topk = known_topk
        self.unknown_score_thresh = unknown_score_thresh
        self.unknown_nms_thresh = unknown_nms_thresh
        self.unknown_topk = unknown_topk
        self.cls_loss_weight = cls_loss_weight

        if "graspnet" in dataset_name:
            known_ids = GRASPNET_KNOWN_IDS
        elif "coco" in dataset_name:
            known_ids = COCO_INVOC_IDS

        meta = MetadataCatalog.get(dataset_name)
        self.class_id, _ = torch.sort(
            torch.tensor(
                [meta.thing_dataset_id_to_contiguous_id[thing_id] for thing_id in known_ids], 
                device='cuda'
            )
        )

        self.id_map = torch.zeros(self.num_classes+1, device='cuda') - 1
        for i, v in enumerate(self.class_id):
            self.id_map[v] = torch.tensor(i, device='cuda')
        self.id_map[self.num_classes] = torch.tensor(self.num_known_classes, device='cuda')

        self.class_id = self.class_id.long()
        self.id_map = self.id_map.long()
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_known_classes"     : cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "unknown_id"            : cfg.MODEL.ROI_HEADS.UNKNOWN_ID,
            "dataset_name"          : cfg.DATASETS.TRAIN[0],
            "input_size"            : cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            "known_score_thresh"    : cfg.MODEL.ROI_HEADS.KNOWN_SCORE_THRESH,
            "known_nms_thresh"      : cfg.MODEL.ROI_HEADS.KNOWN_NMS_THRESH,
            "known_topk"            : cfg.MODEL.ROI_HEADS.KNOWN_TOPK,
            "unknown_score_thresh"  : cfg.MODEL.ROI_HEADS.UNKNOWN_SCORE_THRESH,
            "unknown_nms_thresh"    : cfg.MODEL.ROI_HEADS.UNKNOWN_NMS_THRESH,
            "unknown_topk"          : cfg.MODEL.ROI_HEADS.UNKNOWN_TOPK,
            "cls_loss_weight"       : cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT
        }
    
    def loss(self, dml_features, proposals):
        """
        """
        scores = self.cls_score(dml_features)

        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, self.id_map[gt_classes])

        loss = cross_entropy(scores, self.id_map[gt_classes], reduction="mean")

        return self.cls_loss_weight * loss
    
    def inference(self, fg_instances: List[Instances]):
        """
        """
        results = []
        for fg_instances_per_image in fg_instances:

            known = fg_instances_per_image.pred_classes != self.unknown_id

            known_features = fg_instances_per_image.features[known]
            known_scores = self.cls_score(known_features)
            known_probs = F.softmax(known_scores, dim=-1)

            result_k = fast_rcnn_inference_single_image_known(
                fg_instances_per_image.pred_boxes[known].tensor,
                known_probs,
                fg_instances_per_image.proposal_embedding[known],
                fg_instances_per_image.image_size,
                self.known_score_thresh,
                self.known_nms_thresh,
                self.known_topk
            )
            if not known.all():
                result_unk = fast_rcnn_inference_single_image_unknown(
                    fg_instances_per_image.pred_boxes[~known].tensor,
                    fg_instances_per_image.scores[~known],
                    fg_instances_per_image.proposal_embedding[~known],
                    fg_instances_per_image.image_size,
                    self.unknown_score_thresh,
                    self.unknown_nms_thresh,
                    self.unknown_topk
                )
                res = Instances(fg_instances_per_image.image_size)
                res.pred_boxes = Boxes.cat([result_unk.pred_boxes, 
                                            result_k.pred_boxes])
                res.scores = cat([result_unk.scores, result_k.scores])
                res.pred_classes = cat([result_unk.pred_classes, self.class_id[result_k.pred_classes]])
                res.proposal_embedding = cat([result_unk.proposal_embedding, result_k.proposal_embedding])
            else:
                res = Instances(fg_instances_per_image.image_size)
                res.pred_boxes = result_k.pred_boxes
                res.scores = result_k.scores
                res.pred_classes = self.class_id[result_k.pred_classes]
                res.proposal_embedding = result_k.proposal_embedding
                
            results.append(res)
        
        return results