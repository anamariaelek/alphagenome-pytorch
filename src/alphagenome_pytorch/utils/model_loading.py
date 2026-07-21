"""
Utility for loading AlphaGenome models from either a pretrained .pth or a fine-tuned checkpoint directory (with config.json).

Usage:
    from alphagenome_pytorch.utils.model_loading import load_model_for_inference
    model, cfg = load_model_for_inference(checkpoint_path, device)
"""
import json
from pathlib import Path
import torch

from alphagenome_pytorch.utils.paths import expand_path, expand_paths_in_dict


def _normalize_species_n_conditions(species_n_conditions):
    """Normalize species_n_conditions to dict[int, int].

    Supports dicts (including string keys), list/tuple of pairs, and flat lists.
    """
    if not species_n_conditions:
        return {}

    if isinstance(species_n_conditions, dict):
        return {int(k): int(v) for k, v in species_n_conditions.items()}

    if isinstance(species_n_conditions, (list, tuple)):
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in species_n_conditions):
            return {int(k): int(v) for k, v in species_n_conditions}
        # Fallback: treat as flat list ordered by organism index.
        return {i: int(v) for i, v in enumerate(species_n_conditions)}

    if isinstance(species_n_conditions, int):
        return {0: int(species_n_conditions)}

    raise TypeError(
        "Unsupported species_n_conditions format. Expected dict, list/tuple, or int. "
        f"Got: {type(species_n_conditions).__name__}"
    )

def load_model_for_inference(checkpoint_path, device, strict=True, config_path=None):
    """
    Loads an AlphaGenome model for inference from either:
      - a fine-tuned checkpoint directory (with best_model.pth + config.json)
      - a single .pth file (pretrained or fine-tuned)

    Supports path expansion for:
      - ~ (home directory)
      - $VAR and ${VAR} (environment variables)

    Args:
        checkpoint_path: Checkpoint directory or .pth file.
        device: Target device.
        strict: strict flag passed to load_state_dict for fine-tuned checkpoints.
        config_path: Optional explicit path to the config.json/.yaml to use, for
            checkpoints whose config file isn't named "config.json"/"config.yaml"
            (e.g. a directory containing "config_0-5.json" instead). Overrides
            auto-detection.

    Returns (model, config_dict or None)
    """
    from alphagenome_pytorch import AlphaGenome
    import yaml

    # Expand checkpoint path for ~ and environment variables
    checkpoint_path = expand_path(checkpoint_path)
    p = Path(checkpoint_path)

    if config_path is not None:
        cfg_path = Path(expand_path(config_path))
        if not cfg_path.exists():
            raise FileNotFoundError(f"config_path not found: {cfg_path}")
        if p.is_dir():
            ckpt_file = p / "best_model.pth"
            if not ckpt_file.exists():
                raise FileNotFoundError(f"No best_model.pth found in directory: {p}")
            p = ckpt_file
    # If a directory is given, look for best_model.pth and config.json/yaml
    elif p.is_dir():
        ckpt_file = p / "best_model.pth"
        cfg_json = p / "config.json"
        cfg_yaml = p / "config.yaml"
        if cfg_json.exists():
            cfg_path = cfg_json
        elif cfg_yaml.exists():
            cfg_path = cfg_yaml
        else:
            cfg_path = None
        if not ckpt_file.exists():
            raise FileNotFoundError(f"No best_model.pth found in directory: {p}")
        p = ckpt_file
    else:
        # If a file is given, try to find config.json/yaml in the same directory
        cfg_json = p.parent / "config.json"
        cfg_yaml = p.parent / "config.yaml"
        if cfg_json.exists():
            cfg_path = cfg_json
        elif cfg_yaml.exists():
            cfg_path = cfg_yaml
        else:
            cfg_path = None

    # Try to load config from file if present
    cfg = None
    if cfg_path is not None:
        if cfg_path.suffix == ".json":
            with open(cfg_path) as f:
                cfg = json.load(f)
        elif cfg_path.suffix in (".yaml", ".yml"):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        
        # Expand paths in loaded config
        if cfg is not None:
            path_keys = {
                "genome", "annotation_parquet", "usage_parquet",
                "train_bed", "val_bed", "test_bed", "pretrained_weights"
            }
            cfg = expand_paths_in_dict(cfg, path_keys)

    # Load checkpoint
    ckpt = None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        # If not a torch checkpoint, try pretrained
        model = AlphaGenome.from_pretrained(str(p), device=device)
        return model, None

    # If config not found, try to get from checkpoint
    if cfg is None:
        for key in ["config", "cfg", "config_dict"]:
            if key in ckpt:
                cfg = ckpt[key]
                break
        
        # Expand paths in config from checkpoint
        if cfg is not None:
            path_keys = {
                "genome", "annotation_parquet", "usage_parquet",
                "train_bed", "val_bed", "test_bed", "pretrained_weights"
            }
            cfg = expand_paths_in_dict(cfg, path_keys)

    # If still no config, try to infer if this is a fine-tuned checkpoint
    is_finetuned = False
    if cfg is not None and "species_specs" in cfg:
        is_finetuned = True
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Heuristic: fine-tuned checkpoints usually have model_state_dict
        is_finetuned = True

    if is_finetuned and cfg is None:
        raise RuntimeError(
            f"Checkpoint at {p} looks fine-tuned (state dict has 'model_state_dict', "
            f"'epoch', etc.) but no config.json/config.yaml was found next to it, and "
            f"none of ['config', 'cfg', 'config_dict'] were embedded in the checkpoint. "
            f"Pass the config explicitly via config_path= (e.g. a non-standard filename "
            f"like 'config_0-5.json'); loading it as a plain pretrained checkpoint would "
            f"fail with a confusing state-dict key mismatch."
        )

    if is_finetuned and cfg is not None:
        # Fine-tuned model: reconstruct architecture
        from alphagenome_pytorch.config import DtypePolicy
        from alphagenome_pytorch.extensions.finetuning.heads import create_splice_classification_finetuning_head, create_splice_usage_finetuning_head
        from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads, prepare_for_transfer, TransferConfig
        import torch.nn as nn
        species_specs = cfg["species_specs"]
        num_organisms = max(s["organism_index"] for s in species_specs) + 1
        dict_conds = _normalize_species_n_conditions(cfg.get("species_n_conditions", {}))
        dtype_str = cfg.get("dtype", "bfloat16")
        dtype_policy = DtypePolicy.full_float32() if dtype_str == "float32" else DtypePolicy.mixed_precision()
        model = AlphaGenome(dtype_policy=dtype_policy)
        model = load_trunk(model, cfg["pretrained_weights"], exclude_heads=True)
        model = remove_all_heads(model)
        if cfg.get("mode") == "lora" and cfg.get("lora_rank", 0) > 0:
            lora_targets = [t.strip() for t in cfg["lora_targets"].split(",")]
            lora_cfg = TransferConfig(mode="lora", lora_targets=lora_targets, lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"])
            model = prepare_for_transfer(model, lora_cfg)
        cls_head = create_splice_classification_finetuning_head(num_organisms=num_organisms)
        model.splice_sites_classification_head = cls_head

        # Rebuild per-organism usage heads when present.
        usage_heads = {}
        for org_idx, num_conditions in dict_conds.items():
            if int(num_conditions) > 0:
                usage_heads[int(org_idx)] = create_splice_usage_finetuning_head(
                    n_conditions=int(num_conditions),
                    num_organisms=num_organisms,
                )
        
        # Store as ModuleDict in model.splice_sites_usage_head
        if usage_heads:
            model.splice_sites_usage_head = nn.ModuleDict({str(k): v for k, v in usage_heads.items()})
        else:
            model.splice_sites_usage_head = None

        # Use strict=False by default for fine-tuned checkpoints
        # Strip _orig_mod. prefix that torch.compile adds to state-dict keys
        raw_sd = ckpt["model_state_dict"]
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in raw_sd.items()}
        model.load_state_dict(sd, strict=False if strict is None else strict)

        # Load per-organism usage head state dicts
        usage_heads_state_dicts = ckpt.get("usage_heads_state_dicts", {})
        if model.splice_sites_usage_head is not None and usage_heads_state_dicts:
            for org_key, raw_usd in usage_heads_state_dicts.items():
                org_key = str(org_key)
                if org_key not in model.splice_sites_usage_head:
                    continue
                usd = {
                    (k[len("_orig_mod."): ] if k.startswith("_orig_mod.") else k): v
                    for k, v in raw_usd.items()
                }
                model.splice_sites_usage_head[org_key].load_state_dict(usd, strict=False)
        
        # Backward compatibility: single usage head saved under legacy key (old checkpoints)
        elif "usage_head_state_dict" in ckpt and model.splice_sites_usage_head is not None:
            # Old format: single head, not ModuleDict
            if isinstance(model.splice_sites_usage_head, nn.ModuleDict):
                # Load into first organism if it's now a ModuleDict
                first_org = next(iter(model.splice_sites_usage_head))
                model.splice_sites_usage_head[first_org].load_state_dict(ckpt["usage_head_state_dict"], strict=False)
            else:
                model.splice_sites_usage_head.load_state_dict(ckpt["usage_head_state_dict"], strict=False)

        model.to(device).eval()
        return model, cfg
    else:
        # Pretrained or unknown format: try to load as pretrained
        model = AlphaGenome.from_pretrained(str(p), device=device)
        return model, None

