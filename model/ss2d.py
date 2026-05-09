"""
SS2D — 2-D Selective Scan (mamba-ssm backend)
==============================================
Implements the four-directional scan-expand / S6 / scan-merge pipeline
described in the MBSSNet paper (Fig. 2, Algorithm 1).

Requires:  pip install mamba-ssm einops
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── mamba-ssm selective-scan kernel ──────────────────────────
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[SS2D] WARNING: mamba-ssm not found — falling back to pure-PyTorch (slow).")


# ── Pure-PyTorch fallback for selective scan ─────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False):
    """
    Minimal reference implementation of selective scan.
    u     : (BS, ED, L)     — input sequence
    delta : (BS, ED, L)     — discretization timestep
    A     : (ED, N)         — state matrix
    B     : (BS, N, L)      — input-to-state projection
    C     : (BS, N, L)      — state-to-output projection
    D     : (ED,)           — skip connection
    z     : (BS, ED, L)     — optional gate
    """
    BS, ED, L = u.shape
    N = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    # Discretise:  A is (ED, N), delta is (BS, ED, L)
    # deltaA: (BS, ED, L, N)  =  (BS, ED, L, 1) * (1, ED, 1, N)
    deltaA = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    )  # (BS, ED, L, N)

    # deltaB: (BS, ED, L, N)  =  (BS, ED, L, 1) * (BS, 1, L, N)
    B_t = B.permute(0, 2, 1)                       # (BS, L, N)
    deltaB = delta.unsqueeze(-1) * B_t.unsqueeze(1) # (BS, ED, L, N)

    # Recurrent scan
    h = torch.zeros(BS, ED, N, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        h = deltaA[:, :, i] * h + deltaB[:, :, i] * u[:, :, i].unsqueeze(-1)
        # C[:, :, i] is (BS, N) → need (BS, 1, N) to dot with h (BS, ED, N)
        y_i = (h * C[:, :, i].unsqueeze(1)).sum(dim=-1)   # (BS, ED)
        ys.append(y_i)
    y = torch.stack(ys, dim=-1)  # (BS, ED, L)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)
    if z is not None:
        y = y * F.silu(z)
    return y


# ── Scan helpers ─────────────────────────────────────────────
def scan_expand(x):
    """
    Unfold a 2-D feature map into 4 directional 1-D sequences.
    x : (B, C, H, W)
    Returns: (B, C, 4, L)   where L = H*W
    """
    B, C, H, W = x.shape
    # direction 0: row-major (L→R, T→B)
    d0 = x.reshape(B, C, -1)                               # (B, C, L)
    # direction 1: reversed row-major (R→L, B→T)
    d1 = d0.flip(dims=[-1])
    # direction 2: column-major (T→B, L→R)
    d2 = x.permute(0, 1, 3, 2).reshape(B, C, -1)           # (B, C, L)
    # direction 3: reversed column-major
    d3 = d2.flip(dims=[-1])
    return torch.stack([d0, d1, d2, d3], dim=2)             # (B, C, 4, L)


def scan_merge(ys, H, W):
    """
    Merge 4 directional outputs back into a 2-D feature map.
    ys : (B, C, 4, L)
    Returns : (B, C, H, W)
    """
    B, C, _, L = ys.shape
    y0 = ys[:, :, 0]                                        # (B, C, L)
    y1 = ys[:, :, 1].flip(dims=[-1])
    y2 = ys[:, :, 2].flip(dims=[-1]) if False else ys[:, :, 2]
    # undo column-major
    y2 = ys[:, :, 2].reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
    y3 = ys[:, :, 3].flip(dims=[-1]).reshape(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L)
    merged = y0 + y1 + y2 + y3
    return merged.reshape(B, C, H, W)


# ── SS2D Module ──────────────────────────────────────────────
class SS2D(nn.Module):
    """
    2-D Selective Scan block.

    Parameters
    ----------
    d_model  : input / output channel dimension
    d_state  : SSM state dimension  (N in the paper)
    d_conv   : local DW-conv kernel size
    expand   : inner expansion ratio
    dropout  : dropout rate on output
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, d_model // 16)
        self.K = 4  # scan directions

        # ── projections ──
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=True)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # per-direction projections for Δ, B, C
        self.x_proj = nn.Linear(
            self.d_inner, (self.dt_rank + self.d_state * 2) * self.K, bias=False,
        )
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(self.K)
        ])

        # SSM parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(self.d_inner, 1)          # (D, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # output
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    # ──────────────────────────────────────────────────────────
    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for dt in self.dt_projs:
            nn.init.uniform_(dt.bias, -4.0, 4.0)  # large range so softplus ≈ Δ

    # ──────────────────────────────────────────────────────────
    def forward(self, x):
        """
        x : (B, H, W, C)   — channel-last
        returns : (B, H, W, C)
        """
        B, H, W, C = x.shape
        L = H * W
        K = self.K
        N = self.d_state
        D_inner = self.d_inner
        dt_rank = self.dt_rank

        # 1. input projection  → x_main + gate z
        xz = self.in_proj(x)                                # (B, H, W, 2*D_inner)
        x_main, z = xz.chunk(2, dim=-1)                     # each (B, H, W, D_inner)

        # 2. depth-wise conv (spatial mixing)
        x_main = x_main.permute(0, 3, 1, 2).contiguous()    # (B, D_inner, H, W)
        x_main = self.act(self.conv2d(x_main))               # (B, D_inner, H, W)

        # 3. scan expand  → 4 directions
        xs = scan_expand(x_main)                             # (B, D_inner, 4, L)

        # 4. project to Δ, B, C for ALL directions at once
        #    x_proj: d_inner → (dt_rank + 2*N) * K
        #    We apply it on each (B, L, D_inner) slice, one per direction.
        #    Stack all directions: (B, K, L, D_inner)
        xs_per_dir = xs.permute(0, 2, 3, 1).contiguous()    # (B, K, L, D_inner)
        # Flatten to (B*K, L, D_inner) is wrong because x_proj outputs K projections at once.
        # Instead, we process per-pixel: flatten spatial, project, then split.
        # Use any one direction's features to generate dt/B/C for that direction.

        # Actually, x_proj outputs (dt_rank+2*N)*K per input vector.
        # So we pass each direction's features separately and split the K outputs.
        # Simpler: for each direction k, project xs[:,:,k] → dt_rank+2*N
        # We can batch this efficiently:
        xs_flat = xs_per_dir.reshape(B * K, L, D_inner)     # (B*K, L, D_inner)
        # But x_proj outputs (dt_rank + 2*N) * K dims — too many for per-direction use.
        # The correct approach: each direction gets its OWN dt_rank+2*N slice from x_proj.
        # So we project once with the full x (not per-direction), then split by K.

        # Project from the ORIGINAL (pre-scan) features for each spatial position:
        # Use x_main reshaped to (B, L, D_inner)
        x_flat = x_main.reshape(B, D_inner, L).permute(0, 2, 1)  # (B, L, D_inner)
        dbc_all = self.x_proj(x_flat)                        # (B, L, (dt_rank+2*N)*K)
        dbc_all = dbc_all.reshape(B, L, K, dt_rank + 2 * N) # (B, L, K, dt_rank+2*N)
        dbc_all = dbc_all.permute(2, 0, 1, 3)               # (K, B, L, dt_rank+2*N)

        # 5. per-direction selective scan
        A = -torch.exp(self.A_log.float())                   # (D_inner, N)
        scan_fn = selective_scan_fn if MAMBA_AVAILABLE else selective_scan_ref
        ys_list = []

        for k in range(K):
            u_k = xs[:, :, k].contiguous()                   # (B, D_inner, L)

            dbc_k = dbc_all[k]                               # (B, L, dt_rank+2*N)
            dt_raw_k = dbc_k[:, :, :dt_rank]                 # (B, L, dt_rank)
            B_k = dbc_k[:, :, dt_rank:dt_rank + N]           # (B, L, N)
            C_k = dbc_k[:, :, dt_rank + N:]                  # (B, L, N)

            # dt projection: (B, L, dt_rank) → (B, L, D_inner)
            dt_k = self.dt_projs[k](dt_raw_k)               # (B, L, D_inner)
            dt_k = dt_k.permute(0, 2, 1).contiguous()        # (B, D_inner, L)
            B_k = B_k.permute(0, 2, 1).contiguous()          # (B, N, L)
            C_k = C_k.permute(0, 2, 1).contiguous()          # (B, N, L)

            y_k = scan_fn(
                u_k.float(), dt_k.float(), A, B_k.float(), C_k.float(),
                D=self.D.float(), z=None,
                delta_bias=None, delta_softplus=True,
            )                                                # (B, D_inner, L)
            ys_list.append(y_k)

        ys = torch.stack(ys_list, dim=2)                     # (B, D_inner, 4, L)

        # 6. scan merge
        y = scan_merge(ys, H, W)                             # (B, D_inner, H, W)

        # 7. gate + output projection
        y = y.permute(0, 2, 3, 1).contiguous()               # (B, H, W, D_inner)
        y = self.out_norm(y)
        y = y * self.act(z)                                  # gating
        y = self.out_proj(y)                                 # (B, H, W, C)
        y = self.dropout(y)
        return y

