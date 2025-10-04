# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from ..utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# --------------------- Cosine‑similarity helpers ---------------------

def l2n(x: torch.Tensor, eps: float = 1e-7):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_logits(img_emb: torch.Tensor, txt_bank: torch.Tensor, temperature: float = 50.0):
    return temperature * (img_emb @ txt_bank.t())


# --------------------- Hungarian matching ---------------------

def _gather_diag_boxes(sel_boxes: torch.Tensor, sel_indices: torch.Tensor, diag_indices: torch.Tensor):
    """
    sel_boxes:   [B, S, 4]
    sel_indices: [B, S]  absolute ids of those S boxes
    diag_indices:[B, I]  absolute ids of the I diagonal tokens
    Returns:
      diag_boxes: [B, I, 4]  (zeros when missing)
      present:    [B, I]    (bool mask of which diagonals have a box)
    """
    B, I = diag_indices.shape
    diag_boxes = sel_boxes.new_zeros(B, I, 4)
    present = torch.zeros(B, I, dtype=torch.bool, device=sel_boxes.device)
    for b in range(B):
        # map abs id -> pos in sel_indices[b]
        pos = {int(a): p for p, a in enumerate(sel_indices[b].tolist())}
        for i in range(I):
            abs_id = int(diag_indices[b, i].item())
            if abs_id in pos:
                diag_boxes[b, i] = sel_boxes[b, pos[abs_id]]
                present[b, i] = True
    return diag_boxes, present


"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.
"""
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # --- align boxes to diagonals if indices are provided ---
        if "diag_indices" in outputs and "sel_indices" in outputs and "pred_boxes" in outputs:
            diag_boxes, present = _gather_diag_boxes(
                outputs["pred_boxes"], outputs["sel_indices"], outputs["diag_indices"]
            )  # [B,I,4], [B,I]
            out_bbox = diag_boxes.flatten(0, 1)                  # [B*I,4]
        else:
            out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        if "diag_indices" in outputs and "sel_indices" in outputs:
            miss = (~present).flatten(0, 1).unsqueeze(1).float()            # [B*I, 1]
            cost_bbox  = cost_bbox  + 1e3 * miss                             # big penalty
            cost_giou  = cost_giou  + 1e3 * miss

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

