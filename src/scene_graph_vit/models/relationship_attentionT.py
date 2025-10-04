from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    """y = LN( MLP(x) + x )."""
    def __init__(self, width: int, depth: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(max(depth-1,1))])
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        h = x
        for lin in self.layers:
            h = torch.nn.functional.gelu(lin(h))
        h = self.out(h)
        return self.norm(h + x)

class RelationshipAttention(nn.Module):
    """
    Two-stage hard selection (instances → pairs) with separate subject/object projections.
    """
    def __init__(self, cfg: Dict[str, Any], dim: int):
        super().__init__()
        rel = cfg.get("rel_attn", {})
        self.top_instances = int(rel.get("top_instances", 512))
        self.top_pairs     = int(rel.get("top_pairs", 16384))
        self.mask_self     = bool(rel.get("mask_self_pairs", True))
        depth              = int(rel.get("proj_depth", 2))
        hidmul             = int(rel.get("rel_mlp_multiplier", 2))
        tau_init           = float(rel.get("tau", 6.0))
        self._tau_raw = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))

        self.sub_mlp = ResidualMLP(dim, depth)
        self.obj_mlp = ResidualMLP(dim, depth)
        self.post = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidmul*dim),
            nn.GELU(),
            nn.Linear(hidmul*dim, dim),
        )

    @torch.no_grad()
    def _topk_per_batch(self, scores: torch.Tensor, k: int):
        # scores: [B, M] float32
        k = max(1, min(k, scores.size(1)))
        vals, idx = torch.topk(scores, k=k, dim=1)
        return vals, idx
        
    def forward(self, tokens: torch.Tensor):
        # tokens: [B,N,C]
        B, N, C = tokens.shape
        device = tokens.device
        dtype  = tokens.dtype

        # Learnable temperature parameter (always positive)
        tau = F.softplus(self._tau_raw) + 1e-8

        # 1) separate projections
        # Clone tokens to avoid inference mode issues
        if not tokens.requires_grad:
            tokens = tokens.clone().detach().requires_grad_(True)
        sub = self.sub_mlp(tokens)
        obj = self.obj_mlp(tokens)

        # L2 normalize before computing similarity
        sub_norm = F.normalize(sub, dim=-1, p=2)  # [B,N,C]
        obj_norm = F.normalize(obj, dim=-1, p=2)  # [B,N,C]

        # 2) diagonal (instance) scores and top-I selection
        diag_scores = (sub_norm * obj_norm).sum(-1) * tau  # [B,N]
        I = max(1, min(self.top_instances, N))
        diag_scores_sg = diag_scores.detach()                # no grad through selection
        _vals_I, idx_I = self._topk_per_batch(diag_scores_sg, I)  # [B,I] <- ABSOLUTE indices in [0..N-1]

        # gather top-I
        gather_I = idx_I.unsqueeze(-1).expand(B, I, C)
        sub_I = torch.gather(sub_norm, 1, gather_I)                    # [B,I,C] - already normalized
        obj_I = torch.gather(obj_norm, 1, gather_I)                    # [B,I,C] - already normalized
        
        # For diagonal relation embeddings, use original (non-normalized) features
        sub_I_orig = torch.gather(sub, 1, gather_I)                    # [B,I,C] - original features
        obj_I_orig = torch.gather(obj, 1, gather_I)                    # [B,I,C] - original features
        diag_rel = F.layer_norm(sub_I_orig + obj_I_orig, (C,)).to(dtype)  # [B,I,C]

        # 3) pair scores on I×I with normalization + temperature
        # Compute normalized cosine similarity with temperature scaling
        S = torch.einsum("bic,bjc->bij", sub_I, obj_I) * tau  # [B,I,I]
        
        # More robust self-masking: use -float('inf') instead of -1e9
        if self.mask_self:
            eye = torch.eye(I, device=device, dtype=torch.bool).unsqueeze(0)
            S = S.masked_fill(eye, -float('inf'))

        K = max(1, min(self.top_pairs, I*I))
        flat = S.view(B, I*I)
        flat_sg = flat.detach()
        _, pair_idx = self._topk_per_batch(flat_sg, K)         # [B,K] indices into I×I grid
        rel_scores = flat.gather(1, pair_idx)                  # [B,K] (keep grad through selection)
        
        # CRITICAL FIX: pi and pj are indices into the top-I tokens, NOT absolute indices
        pi = pair_idx // I                                     # [B,K] indices in [0..I-1]
        pj = pair_idx %  I                                     # [B,K] indices in [0..I-1]

        # Post-process to ensure no self-pairs in final selection
        if self.mask_self:
            # Check for any remaining self-pairs and replace them
            self_pair_mask = (pi == pj)  # [B,K] boolean mask
            
            if self_pair_mask.any():
                # For each batch, replace self-pairs with the next best non-self pairs
                for b in range(B):
                    batch_self_mask = self_pair_mask[b]  # [K]
                    if batch_self_mask.any():
                        # Get all non-self pairs from the original scores
                        batch_flat = flat_sg[b]  # [I*I]
                        non_self_mask = torch.ones_like(batch_flat, dtype=torch.bool)
                        
                        # Mark diagonal elements as invalid
                        for i in range(I):
                            diag_idx = i * I + i
                            if diag_idx < len(non_self_mask):
                                non_self_mask[diag_idx] = False
                        
                        # Get top non-self pairs
                        valid_scores = batch_flat[non_self_mask]
                        valid_indices = torch.arange(I*I, device=device)[non_self_mask]
                        
                        if len(valid_scores) > 0:
                            # Sort and get the best available non-self pairs
                            sorted_vals, sorted_idx = torch.sort(valid_scores, descending=True)
                            
                            # Replace self-pairs with best available non-self pairs
                            self_pair_positions = torch.where(batch_self_mask)[0]
                            for idx, pos in enumerate(self_pair_positions):
                                if idx < len(sorted_idx):
                                    replacement_flat_idx = valid_indices[sorted_idx[idx]]
                                    pair_idx[b, pos] = replacement_flat_idx
                                    batch_flat = flat[b]
                                    rel_scores[b, pos] = batch_flat[replacement_flat_idx]
                                    pi[b, pos] = replacement_flat_idx // I
                                    pj[b, pos] = replacement_flat_idx % I

        # 4) *** MAP BACK TO ABSOLUTE TOKEN IDS *** (this was missing!)
        # pi, pj are indices into idx_I [0..I-1], we need absolute token indices [0..N-1]
        subj_idx = torch.gather(idx_I, 1, pi)                          # [B,K] absolute indices
        obj_idx  = torch.gather(idx_I, 1, pj)                          # [B,K] absolute indices
        
        # Also gather the actual embeddings using pi/pj (relative indices)
        sub_sel = torch.gather(sub_I_orig, 1, pi.unsqueeze(-1).expand(B,K,C))
        obj_sel = torch.gather(obj_I_orig, 1, pj.unsqueeze(-1).expand(B,K,C))
        rel = self.post(F.layer_norm(sub_sel + obj_sel, (C,))).to(dtype)  # [B,K,C]

        rel_pairs = torch.stack([subj_idx, obj_idx], dim=-1)           # [B,K,2] NOW ABSOLUTE!
        
        # # DEBUG: Verify absolute IDs are correct
        # print(f"DEBUG RelAttn: diag_indices range: [{idx_I.min()}, {idx_I.max()}]")
        # print(f"DEBUG RelAttn: rel_pairs range: [{rel_pairs.min()}, {rel_pairs.max()}]")
        # print(f"DEBUG RelAttn: rel_pairs sample: {rel_pairs[0, :3].tolist()}")
        
        return {
            "rel_embs": rel,                 # [B,K,C]
            "rel_pairs": rel_pairs,          # [B,K,2] NOW contains absolute token indices in [0..N-1]
            "rel_scores": rel_scores,         # [B,K] (temperature-scaled)
            "diag_rel_embs": diag_rel,       # [B,I,C]
            "diag_indices": idx_I,           # [B,I] absolute indices in [0..N-1]
            "diag_scores": diag_scores,      # [B,N] (temperature-scaled)
        }
