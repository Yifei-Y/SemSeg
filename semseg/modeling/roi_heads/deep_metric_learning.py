import logging
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.layers import nonzero_tuple, cat
from detectron2.data import MetadataCatalog

from semseg.data.graspnet_meta import GRASPNET_KNOWN_IDS
from semseg.data.custom import COCO_INVOC_IDS

logger = logging.getLogger(__name__)


class DML(nn.Module):
    """
    Deep metric learning with double margin loss.
    """
    @configurable
    def __init__(
        self,
        num_classes: int,
        num_known_classes: int,
        unknown_id: int,
        feature_dim: int,
        embedding_dim: int,
        reps_per_class: int,
        distance_type: str,
        alpha: float,
        beta: float,
        loss_weight: float,
        dataset_name: str,
        iou_threshold: float,
        unk_thr: float
    ):
        """
        Args:
            num_classes (int): number of foreground classes (background is not included).
            num_known_classes (int): number of known foreground classes (background is not included).
            unknown_id (int): id for unknown.
            feature_dim (int): dim of RoI feature.
            embedding_dim (int): dim of embedding space in metric learning.
            reps_per_class (int): number of representatives per foreground class.
            distance_type (str): the distance type, current support L1, L2, COS.
            alpha (float): threshold of intra distance.
            beta (float): threshold of inter distance.
            loss_weight (float): weight to use for metric learning loss.
            dataset_name (str): name of dataset used to get the meta data.
            iou_threshold (float): only proposal with iou > iou_threshold used to compute the double margin loss.
            unk_thr (float): the min distance between feature and prototypes > unk_thr, the corresponding proposal is unknown.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_known_classes = num_known_classes
        self.unknown_id = unknown_id
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.reps_per_class = reps_per_class
        self.distance_type = distance_type
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight
        self.unk_thr = unk_thr

        self.encoder = nn.Linear(self.feature_dim, self.embedding_dim, device='cuda')
        nn.init.normal_(self.encoder.weight, std=0.01)
        nn.init.constant_(self.encoder.bias, 0)

        self.decoder = nn.Linear(self.embedding_dim, self.feature_dim, device='cuda')
        nn.init.normal_(self.decoder.weight, std=0.01)
        nn.init.constant_(self.decoder.bias, 0)

        self.representatives = nn.parameter.Parameter(
            torch.zeros(self.num_known_classes * self.reps_per_class, self.embedding_dim)
        )
        nn.init.normal_(self.representatives)

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

        self.iou_threshold = iou_threshold

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes"        : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_known_classes"  : cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "unknown_id"         : cfg.MODEL.ROI_HEADS.UNKNOWN_ID,
            "feature_dim"        : cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            "embedding_dim"      : cfg.MODEL.DML.EMD_DIM,
            "reps_per_class"     : cfg.MODEL.DML.REPS_PER_CLASS,
            "distance_type"      : cfg.MODEL.DML.DISTANCE_TYPE,
            "alpha"              : cfg.MODEL.DML.ALPHA,
            "beta"               : cfg.MODEL.DML.BETA,
            "loss_weight"        : cfg.MODEL.DML.LOSS_WEIGHT, 
            "dataset_name"       : cfg.DATASETS.TRAIN[0],
            "iou_threshold"      : cfg.MODEL.DML.IOU_THRESHOLD,
            "unk_thr"            : cfg.MODEL.DML.UNK_THR,
        }

    def loss(self, roi_features: torch.Tensor, proposals: List[Instances]):
        """
        Double margin loss: L = y_ij * max(Dij-alpha,0) + (1-y_ij) * max(beta-Dij,0).

        Args:
            roi_features (Tensor): shape (num_images * num_samples, feature_dim),
                features after ROI Align and FC.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        
        Returns:
            rec_features (Tensor): shape (num_images * num_samples, feature_dim), decoder output features.
            dml_loss (Tensor): double margin loss.
        """
        # Tensor (num_images * num_samples, embedding_dim)
        emb_features = self.encoder(roi_features)
        new_features = F.normalize(emb_features)
        rec_features = self.decoder(emb_features)
        # Tensor (num_known_classes * reps_per_class, embedding_dim)
        representatives = F.normalize(self.representatives) 

        i = 0
        for instances_per_image in proposals:
            instances_per_image.proposal_embedding = new_features[i:i+len(instances_per_image)]
            i += len(instances_per_image)
        
        ious = (
            cat([p.ious for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )

        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        gt_classes = self.id_map[gt_classes]
        
        fg_inds = nonzero_tuple(
            (gt_classes >= 0) & (gt_classes < self.num_known_classes) & (ious > self.iou_threshold)
        )[0]
        
        new_features = new_features[fg_inds]

        # Tensor (num_fg_samples, num_known_classes * reps_per_class)
        if self.distance_type == 'L1':
            dist = torch.cdist(new_features, representatives, p=1.0)
        elif self.distance_type == 'L2':
            dist = torch.cdist(new_features, representatives)
        elif self.distance_type == 'COS':
            dist = 1 - torch.mm(new_features, representatives.transpose(0,1))
         # Tensor (num_fg_samples, num_known_classes)
        min_dist, _ = torch.min(dist.reshape(-1, self.num_known_classes, self.reps_per_class), dim=2)
        # Tensor (num_fg_samples)
        intra_dist = min_dist[torch.arange(min_dist.shape[0]), gt_classes[fg_inds]] 

        min_dist[torch.arange(min_dist.shape[0]), gt_classes[fg_inds]] = 1000
        inter_dist, _ = torch.min(min_dist, dim=1)

        if self.distance_type == 'L1':
            center_dist = torch.cdist(representatives, representatives, p=1.0)
        elif self.distance_type == 'L2':
            center_dist = torch.cdist(representatives, representatives)
        elif self.distance_type == 'COS':
            center_dist = 1 - torch.mm(representatives, representatives.transpose(0,1))
        center_dist_clone = center_dist.clone()
        for i in range(self.num_known_classes):
            center_dist_clone[i * self.reps_per_class:(i+1)*self.reps_per_class,  i * self.reps_per_class:(i+1)*self.reps_per_class] = 1000
        c_dist, _ = torch.min(center_dist_clone, dim=1)

        dml_loss = torch.sum(torch.max(intra_dist-self.alpha, torch.zeros_like(intra_dist))) + \
            torch.sum(torch.max(self.beta - inter_dist, torch.zeros_like(inter_dist))) + \
            torch.sum(torch.max(self.beta + self.alpha - c_dist, torch.zeros_like(c_dist)))
        
        return rec_features, dml_loss * self.loss_weight / max(gt_classes.numel(), 1.0)

    def inference(self, fg_instances: List[Instances], gt: bool = False):
        """
        Args:
            fg_instances (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections including pred_boxes (Boxes), 
                ious (Tensor) and features (Tensor).

        Returns: 
            results (list[Instances]): add rec_features, pred_classes and proposal_embedding
                to fg_instances, 1000 for unknown.
        """
        representatives = F.normalize(self.representatives)

        results = []
        for fg_instances_per_image in fg_instances:
            features_per_image = fg_instances_per_image.features
            emb_features_per_image = self.encoder(features_per_image)
            new_features_per_image = F.normalize(emb_features_per_image)
            rec_features_per_image = self.decoder(emb_features_per_image)

            if self.distance_type == 'L1':
                dist = torch.cdist(new_features_per_image, representatives, p=1.0)
            elif self.distance_type == 'L2':
                dist = torch.cdist(new_features_per_image, representatives)
            elif self.distance_type == 'COS':
                dist = 1 - torch.mm(new_features_per_image, representatives.transpose(0,1))
            min_dist, _ = torch.min(dist.reshape(-1, self.num_known_classes, self.reps_per_class), dim=2) 
            
            min_dist, min_index = torch.min(min_dist, dim=1)

            unknown = (min_dist > self.unk_thr).nonzero().squeeze()
            min_index = self.class_id[min_index]
            min_index[unknown] = 1000

            fg_instances_per_image.features = rec_features_per_image
            if not gt:
                fg_instances_per_image.pred_classes = min_index

            fg_instances_per_image.proposal_embedding = new_features_per_image

            results.append(fg_instances_per_image)

        return results

    def encode(self, roi_features):
        new_features = F.normalize(self.encoder(roi_features))
        return new_features
