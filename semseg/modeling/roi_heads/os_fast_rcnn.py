# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple, Union
import torch
from torch import nn
from fvcore.nn import smooth_l1_loss

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances

from ..box_regression_w_iou import _dense_box_regression_loss_w_iou

logger = logging.getLogger(__name__)


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    box_features: List[torch.Tensor],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted objectness score for each image.
            Element i has shape (Ri, 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_objectness_score`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        box_features (List[Tensor]): A list of Tensors of box_features used to predict box regression 
            deltas and ious. Element i has shape (Ri, feat_dim), where Ri is the number of predicted objects
            for image i and feat_dim is the feature dimension.
        score_thresh (float): Only return detections with a predicted objectness score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk largest iou detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/ious index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, feats_per_image, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape, feats_per_image in zip(scores, boxes, image_shapes, box_features)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    feats,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on objectness score and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        feats = feats[valid_mask]

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
    feats = feats[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, feats, filter_inds = boxes[keep], scores[keep], feats[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.features = feats
    return result, filter_inds[:, 0]


class OpensetFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. iou
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform: Box2BoxTransform,
        num_classes: int,
        test_objectness_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        box_smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        iou_smooth_l1_beta: float = 0.0,
        iou_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated): box2box transformation type
            num_classes (int): number of foreground classes (background is not included)
            test_objectness_score_thresh (float): threshold to filter predictions results in test.
            test_nms_thresh (float): NMS threshold for prediction results in test.
            test_topk_per_image (int): number of top predictions to produce per image in test.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            iou_smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `iou_reg_loss_type` is "smooth_l1"
            iou_reg_loss_type (str): IoU regression loss type. Current supported losses: "smooth_l1"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_box_reg": applied to box regression loss
                    * "loss_iou_reg": applied to iou regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.iou_pred = nn.Linear(input_size, 1)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.iou_pred.weight, std=0.01)
        for l in [self.bbox_pred, self.iou_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.box_smooth_l1_beta = box_smooth_l1_beta
        self.test_objectness_score_thresh = test_objectness_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        self.iou_smooth_l1_beta = iou_smooth_l1_beta
        self.iou_reg_loss_type = iou_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_box_reg": loss_weight, "loss_iou_reg": loss_weight}
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"                        : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg"              : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "box_smooth_l1_beta"                 : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_objectness_score_thresh"       : cfg.MODEL.ROI_HEADS.OBJ_SCORE_THRESH_TEST,
            "test_nms_thresh"                    : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"                : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"                  : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "iou_smooth_l1_beta"                 : cfg.MODEL.ROI_BOX_HEAD.IOU_SMOOTH_L1_BETA,
            "iou_reg_loss_type"                  : cfg.MODEL.ROI_BOX_HEAD.IOU_REG_LOSS_TYPE,
            "loss_weight"                        : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
                                                    "loss_iou": cfg.MODEL.ROI_BOX_HEAD.IOU_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            proposal_deltas (Tensor): bounding box regression deltas for each box. Shape is shape (N,Kx4),
                or (N,4) for class-agnostic regression.
            pred_iou (Tensor): ou prediction for each box. Shape is (N, 1) dd
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        pred_iou = self.iou_pred(x).sigmoid()
        return proposal_deltas, pred_iou

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``ious`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        # Tensor (num_images*num_samples, 4) or (numimages*num_samples, 1)
        proposal_deltas, pred_iou = predictions

        # Tensor: (num_images * num_samples)
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        gt_iou = (
            cat([p.ious for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device) 

        losses = {
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
            "loss_iou": self.iou_loss(pred_iou, gt_iou, gt_classes)
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss_w_iou(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.box_smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def iou_loss(self, pred_iou, gt_iou, gt_classes):
        """
        IoU regression loss.

        Args: 
            pred_iou (Tensor): shape (num_images * num_samples, 1), IoU prediction
            gt_iou (list[Tensor]): length #images list, element i is length num_samples Tensor containing the ground truth IoU
            gt_classes (Tensor): length #images * num_samples, the gt class label of each proposal
        
        Returns:
            Tensor: IoU loss
        """
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        fg_pred_iou = pred_iou.squeeze()[fg_inds]
        fg_gt_iou = gt_iou[fg_inds]
        loss_iou = smooth_l1_loss(fg_pred_iou, fg_gt_iou, beta=self.iou_smooth_l1_beta, reduction='sum')
        
        return loss_iou / max(gt_classes.numel(), 1.0)

    def inference(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        proposals: List[Instances],
        box_features: torch.Tensor
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes``, ``objectness_logits`` 
                and ``image_size`` field is expected.

        Returns:
            see `find_top_proposals`
        """
        boxes = self.predict_boxes(predictions, proposals)
        score = self.predict_objectness_score(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        num_prop_per_image = [len(p) for p in proposals]
        box_features = box_features.split(num_prop_per_image)
        return fast_rcnn_inference(
            boxes,
            score,
            image_shapes,
            box_features,
            self.test_objectness_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image
        )

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        proposal_deltas, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_objectness_score(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``objectness_logits`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted objectness score (sqrt(ios*centerness)) for each image.
                Element i has shape (Ri, 1), where Ri is the number of proposals for image i.
        """
        _, ious = predictions
        centerness = cat([p.objectness_logits for p in proposals]).unsqueeze(1)
        scores = torch.sqrt(ious * centerness)
        num_prop_per_image = [len(p) for p in proposals]
        return scores.split(num_prop_per_image)
