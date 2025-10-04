# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..utils import box_ops
from ..utils.misc import accuracy, get_world_size, is_dist_avail_and_initialized
from .matcher import HungarianMatcher, build_matcher, cosine_logits, l2n


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    


class OpenVocabSetCriterion(SetCriterion):

    def __init__(self, obj_text_bank: torch.Tensor, pred_text_bank: torch.Tensor,
                 matcher, temp_obj: float = 0.07, temp_pred: float = 0.07,
                 weight_dict=None, losses=None):
        
        num_classes = obj_text_bank.size(0)
        if weight_dict is None:
            weight_dict = {"loss_obj_bce":1, "loss_pred_bce":1, "loss_bbox":1, "loss_giou":1, "loss_ra_bce":1}
        if losses is None:
            losses = ["ov_labels", "boxes", "predicates", "ra"]

        # eos_coef kept for base init but not used in OV BCE
        super().__init__(num_classes=num_classes, matcher=matcher,
                         weight_dict=weight_dict, eos_coef=0.1, losses=losses)
        self.register_buffer("obj_bank", obj_text_bank, persistent=False)   # [C_obj, D]
        self.register_buffer("pred_bank", pred_text_bank, persistent=False) # [C_pred, D]
        self.temp_obj = float(temp_obj)
        self.temp_pred = float(temp_pred)

    # ---------- small helper ----------
    @staticmethod
    @torch.no_grad()
    def _diag_abs_to_sel_pos(diag_indices_b: torch.Tensor, sel_indices_b: torch.Tensor):
        # dict mapping; robust for large sparse absolute ids
        m = {int(abs_id.item()): -1 for abs_id in diag_indices_b}
        for pos, abs_id in enumerate(sel_indices_b.tolist()):
            if abs_id in m:
                m[abs_id] = pos
        return m

    # ---------- OV labels (objects) ----------
    def loss_ov_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Build object logits from diag_rel_embs against obj_text_bank and run BCE
        over all diagonals: matched → positive for the GT class; unmatched → all-zero (negative).
        """
        diag_rel = outputs["diag_rel_embs"]      # [B,I,D]
        obj_logits = cosine_logits(l2n(diag_rel), l2n(self.obj_bank), temperature=self.temp_obj)  # [B,I,C_obj]

        B, I, C_obj = obj_logits.shape
        Y_obj = obj_logits.new_zeros((B, I, C_obj))
        for b, (src_i, tgt_j) in enumerate(indices):
            if len(src_i) == 0: 
                continue
            lb = targets[b]["labels"][tgt_j.long()].long()   # [M]
            Y_obj[b, src_i.long(), lb] = 1.0

        loss_obj_bce = F.binary_cross_entropy_with_logits(obj_logits, Y_obj)

        losses = {"loss_obj_bce": loss_obj_bce}
        if log and sum(len(t["labels"]) for t in targets) > 0:
            # optional: report class error on matched positions (approx)
            idx = self._get_src_permutation_idx(indices)
            with torch.no_grad():
                # take argmax over classes on matched diagonals
                pred_on_match = obj_logits[idx]                  # [M, C_obj]
                tgt_classes   = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                acc = accuracy(pred_on_match, tgt_classes)[0]
                losses["class_error"] = 100 - acc
        # cache to avoid recomputation in other terms
        outputs["_obj_logits"] = obj_logits
        return losses

    # ---------- Boxes (matched diagonals only) ----------
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Reuse DETR's L1 + gIoU but the per-diagonal box is taken from sel_indices
        by matching absolute token ids (diag_indices ↔ sel_indices).
        """
        diag_idx    = outputs["diag_indices"]    # [B,I]
        sel_indices = outputs["sel_indices"]     # [B,S]
        sel_boxes   = outputs["pred_boxes"]      # [B,S,4]

        l1_losses, giou_losses = [], []

        for b, (src_i, tgt_j) in enumerate(indices):
            if len(src_i) == 0: 
                continue

            pos_map = self._diag_abs_to_sel_pos(diag_idx[b], sel_indices[b])
            src_i = src_i.long(); tgt_j = tgt_j.long()

            pred_batch, tgt_batch = [], []
            for si, tj in zip(src_i.tolist(), tgt_j.tolist()):
                abs_id = int(diag_idx[b, si].item())
                p = pos_map.get(abs_id, -1)
                if p != -1:
                    pred_batch.append(sel_boxes[b, p])
                    tgt_batch.append(targets[b]["boxes"][tj])

            if pred_batch:
                pred_boxes = torch.stack(pred_batch)    # [m,4]
                tgt_boxes  = torch.stack(tgt_batch)     # [m,4]

                l1 = F.l1_loss(pred_boxes, tgt_boxes, reduction="none").mean(dim=1).mean()
                l1_losses.append(l1)

                giou = 1.0 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(pred_boxes),
                    box_ops.box_cxcywh_to_xyxy(tgt_boxes)
                )).mean()
                giou_losses.append(giou)

        loss_bbox = torch.stack(l1_losses).mean() if l1_losses else sel_boxes.new_tensor(0.0, requires_grad=True)
        loss_giou = torch.stack(giou_losses).mean() if giou_losses else sel_boxes.new_tensor(0.0, requires_grad=True)

        # Normalize
        loss_bbox = loss_bbox / max(1.0, num_boxes)
        loss_giou = loss_giou / max(1.0, num_boxes)
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    # ---------- Predicates (pairs) ----------
    def loss_predicates(self, outputs, targets, indices, num_boxes):
        """
        Multi-label BCE on rel_embs against pred_text_bank. We build a [B,K,C_pred]
        multi-hot target by mapping GT box indices through the diagonal matches
        to absolute token ids, then locating the corresponding rows in rel_pairs.
        """
        pair_rel    = outputs["rel_embs"]     # [B,K,D]
        rel_pairs   = outputs["rel_pairs"]    # [B,K,2] absolute ids
        pred_logits = cosine_logits(l2n(pair_rel), l2n(self.pred_bank), temperature=self.temp_pred)

        B, K, C_pred = pred_logits.shape
        Y = pred_logits.new_zeros((B, K, C_pred))
        diag_idx = outputs["diag_indices"]   # [B,I]

        any_exact = False
        total_relations = 0
        matched_relations = 0

        for b in range(B):
            if "relations" not in targets[b] or targets[b]["relations"].numel() == 0:
                continue

            gt = targets[b]["relations"].long()  # [R,3]
            total_relations += len(gt)
            # print(f"Batch {b}: Processing {len(gt)} relations")

            # Map GT object idx -> abs token via Hungarian matches
            gt2abs = pred_logits.new_full((targets[b]["boxes"].size(0),), -1, dtype=torch.long)
            src_i, tgt_j = indices[b]
            # print(f"Batch {b}: Hungarian matches: src_i={len(src_i)}, tgt_j={len(tgt_j)}")
            
            if len(src_i) > 0:
                gt2abs[tgt_j.long()] = diag_idx[b][src_i.long()]
                # print(f"Batch {b}: gt2abs mapping: {gt2abs.tolist()[:10]}...")

            s_abs = gt2abs[gt[:,0]]
            o_abs = gt2abs[gt[:,1]]
            valid = (s_abs >= 0) & (o_abs >= 0)
            # print(f"Batch {b}: Valid relations after gt2abs mapping: {valid.sum()}/{len(valid)}")
            
            if not valid.any():
                # print(f"Batch {b}: No valid relations after mapping")
                continue

            # Find exact matches in rel_pairs
            s_abs = s_abs[valid]; o_abs = o_abs[valid]; cls = gt[valid,2]
            pairs_b = rel_pairs[b]  # [K,2]
            
            # # Debug: Print sample values to verify matching
            # print(f"Batch {b}: Sample rel_pairs: {pairs_b[:5].tolist()}")
            # print(f"Batch {b}: GT pairs (s_abs, o_abs): {list(zip(s_abs.tolist(), o_abs.tolist()))[:3]}")
            
            eq_s = (pairs_b[:,0:1] == s_abs.unsqueeze(0))
            eq_o = (pairs_b[:,1:2] == o_abs.unsqueeze(0))
            k_idx, j_idx = (eq_s & eq_o).nonzero(as_tuple=True)

            if k_idx.numel() > 0:
                Y[b, k_idx, cls[j_idx]] = 1.0
                any_exact = True
                matched_relations += k_idx.numel()
                # print(f"Batch {b}: Found {k_idx.numel()} exact matches")
            # else:
                # print(f"Batch {b}: No exact matches found")

        # print(f"Total matched relations: {matched_relations}/{total_relations}")
        loss_pred_bce = F.binary_cross_entropy_with_logits(pred_logits, Y)
        outputs["_pred_logits"] = pred_logits
        return {"loss_pred_bce": loss_pred_bce}


    # ---------- RA self-supervision ----------
    def loss_ra(self, outputs, targets, indices, num_boxes):
        """
        Binary CE between rel_scores and the max class probability of the same pair:
          - diagonal pairs: max over OBJECT classes (from _obj_logits)
          - off-diagonal:   max over PREDICATE classes (from _pred_logits)
        """
        rel_scores = outputs["rel_scores"]   # [B,K]
        rel_pairs  = outputs["rel_pairs"]    # [B,K,2]
        obj_logits = outputs.get("_obj_logits", None)
        pred_logits= outputs.get("_pred_logits", None)

        # if caches missing (e.g., called out of order), recompute quickly
        if obj_logits is None:
            diag_rel = outputs["diag_rel_embs"]
            obj_logits = cosine_logits(l2n(diag_rel), l2n(self.obj_bank), temperature=self.temp_obj)
        if pred_logits is None:
            pair_rel  = outputs["rel_embs"]
            pred_logits = cosine_logits(l2n(pair_rel), l2n(self.pred_bank), temperature=self.temp_pred)

        with torch.no_grad():
            pred_max = torch.sigmoid(pred_logits).amax(dim=-1)   # [B,K]
            is_diag  = (rel_pairs[...,0] == rel_pairs[...,1])    # [B,K]

            # map diag absolute id -> diagonal index i to pull obj probs
            diag_idx  = outputs["diag_indices"]                  # [B,I]
            obj_probs = torch.sigmoid(obj_logits).amax(dim=-1)   # [B,I]
            obj_for_pair = torch.zeros_like(pred_max)

            B, K = pred_max.shape
            I = diag_idx.size(1)
            for b in range(B):
                diag_map = {}
                for i in range(I):
                    abs_id = int(diag_idx[b, i].item())
                    diag_map[abs_id] = i
                
                for k in range(K):
                    if is_diag[b, k]:
                        s_abs = int(rel_pairs[b, k, 0].item())
                        if s_abs in diag_map:
                            i = diag_map[s_abs]
                            obj_for_pair[b, k] = obj_probs[b, i]

            ra_target = torch.where(is_diag, obj_for_pair, pred_max)  # [B,K]

        # loss_ra_bce = F.binary_cross_entropy_with_logits(rel_scores, ra_target)
        
        ra_target = ra_target.clamp(0.02, 0.98)
        loss_ra_bce = F.binary_cross_entropy_with_logits(rel_scores, ra_target, reduction="mean")
        # Add small regularization to prevent runaway behavior
        reg_term = 0.001 * torch.mean(torch.abs(rel_scores))
        loss_ra_bce = loss_ra_bce + reg_term

        return {"loss_ra_bce": loss_ra_bce}

    # ---------- override get_loss to route OV names ----------
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "ov_labels":   self.loss_ov_labels,   # object BCE via text bank
            "boxes":       self.loss_boxes,       # L1 + gIoU on matched diagonals
            "predicates":  self.loss_predicates,  # predicate BCE
            "ra":          self.loss_ra,          # RA self-supervision
        }
        assert loss in loss_map, f"Unknown loss '{loss}'"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        Expected keys in `outputs`:
          - diag_rel_embs [B,I,D], diag_indices [B,I]
          - rel_embs [B,K,D], rel_pairs [B,K,2], rel_scores [B,K]
          - pred_boxes [B,S,4], sel_indices [B,S]
        """
        # Build object logits once for matching (the method also caches them for losses)
        diag_rel = outputs["diag_rel_embs"]
        obj_logits = cosine_logits(l2n(diag_rel), l2n(self.obj_bank), temperature=self.temp_obj)

        # Hungarian matching over diagonals using OV logits + aligned boxes
        match_in = {
            "pred_logits": obj_logits,               # [B,I,C_obj]
            "pred_boxes":  outputs["pred_boxes"],    # [B,S,4]
            "sel_indices": outputs["sel_indices"],   # [B,S]
            "diag_indices":outputs["diag_indices"],  # [B,I]
        }
        indices = self.matcher(match_in, targets)

        # DETR-style normalization by total GT boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=diag_rel.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute requested losses
        losses = {}
        # seed caches so loss_ra/loss_predicates can reuse
        outputs["_obj_logits"] = obj_logits
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                # compute aux obj logits for matcher? Typically we reuse main matches; but paper trains cls on aux only.
                aux["diag_rel_embs"] = aux["diag_rel_embs"]
                aux_obj_logits = cosine_logits(l2n(aux["diag_rel_embs"]), l2n(self.obj_bank), temperature=self.temp_obj)
                aux["_obj_logits"] = aux_obj_logits
                # reuse same indices for aux OR recompute (choose one); we recompute for symmetry:
                aux_match_in = {
                    "pred_logits": aux_obj_logits,
                    "pred_boxes":  aux["pred_boxes"],
                    "sel_indices": aux["sel_indices"],
                    "diag_indices":aux["diag_indices"],
                }
                aux_indices = self.matcher(aux_match_in, targets)

                for loss in self.losses:
                    # disable logging duplicates where needed
                    ldict = self.get_loss(loss, aux, targets, aux_indices, num_boxes)
                    ldict = {f"{k}_{i}": v for k, v in ldict.items()}
                    losses.update(ldict)

        # combine into a single total for convenience
        total = 0.0
        for k, w in self.weight_dict.items():
            if k in losses:
                total = total + w * losses[k]
        losses["loss_total"] = total
        return losses