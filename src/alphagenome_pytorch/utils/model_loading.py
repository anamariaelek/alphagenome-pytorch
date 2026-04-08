"""
Utility for loading AlphaGenome models from either a pretrained .pth or a fine-tuned checkpoint directory (with config.json).

Usage:
    from alphagenome_pytorch.utils.model_loading import load_model_for_inference
    model, cfg = load_model_for_inference(checkpoint_path, device)
"""
import sys
import json
from pathlib import Path
import torch

def load_model_for_inference(checkpoint_path, device, strict=True):
    """
    Loads an AlphaGenome model for inference from either:
      - a fine-tuned checkpoint directory (with best_model.pth + config.json)
      - a single .pth file (pretrained or fine-tuned)
    Returns (model, config_dict or None)
    """
    from alphagenome_pytorch import AlphaGenome
    import yaml
    p = Path(checkpoint_path)

    # If a directory is given, look for best_model.pth and config.json/yaml
    if p.is_dir():
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

    # Load checkpoint
    ckpt = None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        # If not a torch checkpoint, try pretrained
        model = AlphaGenome.from_pretrained(str(p), device=device)
        return model, None

    # If config not found, try to get from checkpoint
    if cfg is None:
        for key in ["config", "cfg", "config_dict"]:
            if key in ckpt:
                cfg = ckpt[key]
                break

    # If still no config, try to infer if this is a fine-tuned checkpoint
    is_finetuned = False
    if cfg is not None and "species_specs" in cfg:
        is_finetuned = True
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Heuristic: fine-tuned checkpoints usually have model_state_dict
        is_finetuned = True

    if is_finetuned and cfg is not None:
        # Fine-tuned model: reconstruct architecture
        from alphagenome_pytorch.config import DtypePolicy
        from alphagenome_pytorch.extensions.finetuning.heads import create_splice_classification_finetuning_head
        from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads, prepare_for_transfer, TransferConfig
        species_specs = cfg["species_specs"]
        num_organisms = max(s["organism_index"] for s in species_specs) + 1
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
        # Use strict=False by default for fine-tuned checkpoints
        # Strip _orig_mod. prefix that torch.compile adds to state-dict keys
        raw_sd = ckpt["model_state_dict"]
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in raw_sd.items()}
        model.load_state_dict(sd, strict=False if strict is None else strict)
        model.to(device).eval()
        return model, cfg
    else:
        # Pretrained or unknown format: try to load as pretrained
        model = AlphaGenome.from_pretrained(str(p), device=device)
        return model, None
