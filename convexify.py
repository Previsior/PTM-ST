"""
example usage:
python convexify.py --start 0 --end 10 --normal_dir ./buffer/flickr/nfnet_bert/InfoNCE/normal
"""

import os
import glob
import torch
import argparse


def convexify(traj):
    """
    Given one trajectory tau_mtt  (list of timestamps, each timestamp is
    a list of parameter tensors), return tau_conv.

    Layer-wise beta_l(t) is computed from accumulated layer-wise L2 norms.
    theta_l(t) = (1-beta_l(t))*theta_l(0) + beta_l(t)*theta_l(K).
    """
    traj = traj[START:END]
    K = len(traj) - 1                 # K+1 timestamps in total
    n_layers = len(traj[0])

    # pre-allocate tensor to hold ||theta^{p+1}_l - theta^{p}_l||2
    norms = torch.zeros(K, n_layers)

    # compute layer-wise step lengths
    for p in range(K):
        for l in range(n_layers):
            diff = traj[p + 1][l] - traj[p][l]
            norms[p, l] = diff.norm().item()   # keep on CPU

    total = norms.sum(0)              # sum_{p=0}^{K-1} ||*|| per layer
    cumsum = norms.cumsum(0)          # cumulative sum over p

    tau_conv = []
    theta0 = traj[0]
    thetaK = traj[-1]

    for t in range(K + 1):
        if t == 0:
            tau_conv.append([p.clone().cpu() for p in theta0])
            continue
        if t == K:
            tau_conv.append([p.clone().cpu() for p in thetaK])
            continue

        beta = torch.where(total > 0, cumsum[t - 1] / total, torch.zeros_like(total))  # shape (n_layers,)

        ts_params = []
        for l in range(n_layers):
            b = beta[l].item()
            new_param = (1.0 - b) * theta0[l] + b * thetaK[l]
            ts_params.append(new_param.clone().cpu())

        tau_conv.append(ts_params)

    return tau_conv


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
        print(f"[OK] Saved convexified trajectory -> {dst_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convexify parameter trajectories.")
    parser.add_argument("--normal_dir", type=str, default="./buffer/flickr/nfnet_bert/InfoNCE/normal", help="Source directory")
    parser.add_argument("--start", type=int, default=0, help="Start index for slicing")
    parser.add_argument("--end", type=int, default=8, help="End index for slicing")
    
    args = parser.parse_args()

    NORMAL_DIR = args.normal_dir
    START = args.start
    END = args.end
    
    parent_dir = os.path.dirname(NORMAL_DIR.rstrip('/'))
    CONV_DIR = os.path.join(parent_dir, f"convexified_{START}_{END}")
    os.makedirs(CONV_DIR, exist_ok=True)
    
    print(f"Processing from: {NORMAL_DIR}")
    print(f"Slice range: {START} to {END}")
    print(f"Saving to: {CONV_DIR}")

    process_dir("img_replay_buffer_")
    process_dir("txt_replay_buffer_")