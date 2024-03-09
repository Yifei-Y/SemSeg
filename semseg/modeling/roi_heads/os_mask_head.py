# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.modeling.roi_heads import BaseMaskRCNNHead, ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss, mask_rcnn_inference


@ROI_MASK_HEAD_REGISTRY.register()
class ClsSemModMaskHead(BaseMaskRCNNHead):
    """
    Class semantics modulation mask head.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, conv_dims, mask_emb_dim, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            mask_emb_dim (int): the dimension of the mask embedding.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv = nn.Sequential()
        self.conv_list = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.conv.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_list.append(conv)
            cur_channels = conv_dim

        deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.conv.add_module("deconv", deconv)
        self.conv_list.append(deconv)
        self.conv.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        post_conv = Conv2d(
            cur_channels,
            mask_emb_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv.add_module("post_conv", post_conv)
        self.conv_list.append(post_conv)

        for layer in self.conv_list:
            weight_init.c2_msra_fill(layer)
        
        self.mask_embed = nn.Sequential(
            nn.Linear(cur_channels, cur_channels), nn.ReLU(inplace=True),
            nn.Linear(cur_channels, mask_emb_dim)
        )
        for layer in self.mask_embed:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
            mask_emb_dim=cfg.MODEL.ROI_MASK_HEAD.MASK_EMB_DIM,
        )
        return ret

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`, (num_proposals, 256, 14, 14).
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        # (num_proposals, 256)
        proposal_embedding = (
            cat([i.proposal_embedding for i in instances], dim=0)
        )
        # (num_proposals, mask_emb_dim)
        mask_embed = self.mask_embed(proposal_embedding)
        # (num_proposals, mask_emb_dim, 28, 28)
        x = self.conv(x)
        # (num_proposals, 1, 28, 28)
        x = torch.sum(mask_embed.unsqueeze(-1).unsqueeze(-1) * x, dim=1, keepdim=True)
        
        if self.training:
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(x, instances)
            return instances
