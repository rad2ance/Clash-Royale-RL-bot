from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from crbot.data import BehaviorCloningDataset, list_episode_files
from crbot.models import BcPolicy


def apply_action_mask(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # Large negative value rather than -inf keeps CE numerically stable.
    return logits.masked_fill(~masks.bool(), -1e9)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        if len(batch) == 3:
            obs, actions, masks = batch
            masks = masks.to(device)
        else:
            obs, actions = batch
            masks = None
        obs = obs.to(device)
        actions = actions.to(device)
        logits = model(obs)
        if masks is not None:
            logits = apply_action_mask(logits, masks)
        loss = ce(logits, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = obs.size(0)
        total_loss += float(loss.item()) * bs
        total += bs
    return total_loss / max(total, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        if len(batch) == 3:
            obs, actions, masks = batch
            masks = masks.to(device)
        else:
            obs, actions = batch
            masks = None
        obs = obs.to(device)
        actions = actions.to(device)
        logits = model(obs)
        if masks is not None:
            logits = apply_action_mask(logits, masks)
        loss = ce(logits, actions)
        preds = torch.argmax(logits, dim=1)
        bs = obs.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((preds == actions).sum().item())
        total += bs
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavior cloning policy.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mask-actions", action="store_true", help="Use per-sample action masks when available.")
    parser.add_argument("--out", type=str, default="checkpoints/bc_policy.pt")
    args = parser.parse_args()

    files = list_episode_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No .npz trajectories found in: {args.data_dir}")

    dataset = BehaviorCloningDataset(files)
    obs_dim = int(dataset.observations.shape[1])
    n_actions = int(dataset.actions.max()) + 1
    if dataset.action_masks is not None:
        n_actions = max(n_actions, int(dataset.action_masks.shape[1]))
    use_masks = bool(args.mask_actions and dataset.action_masks is not None)
    if args.mask_actions and dataset.action_masks is None:
        raise ValueError("Requested --mask-actions but loaded dataset has no action_masks.")
    print(f"[data] transitions={len(dataset)} obs_dim={obs_dim} n_actions={n_actions} mask_actions={use_masks}")

    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    obs_t = torch.from_numpy(dataset.observations)
    act_t = torch.from_numpy(dataset.actions).long()
    if use_masks:
        mask_t = torch.from_numpy(dataset.action_masks).bool()
        train_ds = TensorDataset(obs_t[train_idx], act_t[train_idx], mask_t[train_idx])
        val_ds = TensorDataset(obs_t[val_idx], act_t[val_idx], mask_t[val_idx])
    else:
        train_ds = TensorDataset(obs_t[train_idx], act_t[train_idx])
        val_ds = TensorDataset(obs_t[val_idx], act_t[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BcPolicy(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, device)
        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()
    torch.save(
        {
            "state_dict": best_state,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden_dim": args.hidden_dim,
            "masked_actions": use_masks,
        },
        out_path,
    )
    print(f"[done] saved BC checkpoint: {out_path.resolve()}")


if __name__ == "__main__":
    main()
