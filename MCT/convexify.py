import os
import glob
import torch
import numpy as np


def interp_params(traj_dict, c):
    """
    traj_dict : {'theta0', 'thetaK', 'beta'} ; beta shape (L, K+1)
    c         : float  in [0,K]
    return    : list[layer Tensor]  (ˆθ(c))
    """
    beta_mat = traj_dict['beta']   # (L, K+1)

    t0, t1 = int(np.floor(c)), int(np.ceil(c))
    eta = c - t0
    beta_hat = (1 - eta) * beta_mat[:, t0] + eta * beta_mat[:, t1]   # (L,)

    θ0, θK = traj_dict['theta0'], traj_dict['thetaK']
    out = [(1 - b) * p0 + b * pK for b, p0, pK in zip(beta_hat, θ0, θK)]
    return out

# ---------------------------------------------------------------------
NORMAL_DIR = "./buffer/cc3m/nfnet_bert/InfoNCE/normal"
START = 0
END = 10
CONV_DIR   = f"./buffer/cc3m/nfnet_bert/InfoNCE/convexified_{START}_{END}"
# ---------------------------------------------------------------------

os.makedirs(CONV_DIR, exist_ok=True)


def convexify(traj):
    """
    Given one trajectory τ_mtt  (list of timestamps, each timestamp is
    a list of parameter tensors), return τ_conv.

    Layer-wise β_l(t) is computed from accumulated layer-wise L2 norms.
    ˆθ_l(t) = (1-β_l(t))·θ_l(0) + β_l(t)·θ_l(K).
    """
    traj = traj[START:END]
    K = len(traj) - 1                 # K+1 timestamps in total
    n_layers = len(traj[0])

    # pre‑allocate tensor to hold ‖θ^{p+1}_l − θ^{p}_l‖₂
    norms = torch.zeros(K, n_layers)

    # compute layer‑wise step lengths
    for p in range(K):
        for l in range(n_layers):
            diff = traj[p + 1][l] - traj[p][l]
            norms[p, l] = diff.norm().item()   # keep on CPU

    total = norms.sum(0)              # ∑_{p=0}^{K‑1} ‖·‖ per layer
    cumsum = norms.cumsum(0)          # cumulative sum over p

    τ_conv = []
    θ0 = traj[0]
    θK = traj[-1]

    for t in range(K + 1):
        if t == 0:
            τ_conv.append([p.clone().cpu() for p in θ0])
            continue
        if t == K:
            τ_conv.append([p.clone().cpu() for p in θK])
            continue

        beta = torch.where(total > 0, cumsum[t - 1] / total, torch.zeros_like(total))  # shape (n_layers,)

        ts_params = []
        for l in range(n_layers):
            b = beta[l].item()
            new_param = (1.0 - b) * θ0[l] + b * θK[l]
            ts_params.append(new_param.clone().cpu())

        τ_conv.append(ts_params)

    return τ_conv

def convexify_beta(traj):
    traj = traj[START:END]
    K, n_layers = len(traj)-1, len(traj[0])
    seg_norm = torch.zeros(n_layers, K)          # (L,K)
    for p in range(K):
        for l in range(n_layers):
            seg_norm[l, p] = (traj[p+1][l] - traj[p][l]).norm().float()

    total = seg_norm.sum(1, keepdim=True)        # (L,1)
    cum = torch.cat([torch.zeros(n_layers, 1), seg_norm.cumsum(1)], dim=1)  # (L,K+1)
    beta = torch.where(total > 0, cum / total, torch.zeros_like(cum))        # (L,K+1)

    return {
        'theta0': [p.cpu() for p in traj[0]],
        'thetaK': [p.cpu() for p in traj[-1]],
        'beta'  : beta.cpu()                    # layer-wise
    }


def process_dir(prefix):
    """
    Convexify every *.pt file starting with `prefix` in NORMAL_DIR and
    write to CONV_DIR with the same filename.
    """
    pattern = os.path.join(NORMAL_DIR, f"{prefix}*.pt")
    for src_path in sorted(glob.glob(pattern)):
        trajectories = torch.load(src_path, map_location="cpu")   # list
        new_trajectories = [convexify(traj) for traj in trajectories]

        dst_path = os.path.join(CONV_DIR, os.path.basename(src_path))
        torch.save(new_trajectories, dst_path)
        print(f"[✓] Saved convexified trajectory → {dst_path}")

def process_dir_beta(prefix):
    for src in glob.glob(os.path.join(NORMAL_DIR, f"{prefix}*.pt")):
        new_buf = [convexify_beta(traj) for traj in torch.load(src, map_location='cpu')]
        torch.save(new_buf, os.path.join(CONV_DIR, os.path.basename(src)))
        print("✓ convexified →", os.path.basename(src))


if __name__ == "__main__":
    # image‑side trajectories
    process_dir("img_replay_buffer_")
    # process_dir_beta("img_replay_buffer_")
    # text‑side trajectories
    process_dir("txt_replay_buffer_")
    # process_dir_beta("txt_replay_buffer_")
