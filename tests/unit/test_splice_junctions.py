"""Tests for splice-junction targets, head factory and loss."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from convert_splice_junctions_to_parquet import (  # noqa: E402
    adjust_junction_positions,
    convert_spliser_to_junction_parquet,
    extract_junctions,
)

from alphagenome_pytorch.extensions.finetuning.heads import create_splice_junction_finetuning_head
from alphagenome_pytorch.extensions.finetuning.splice_datasets import (
    SpliceJunctionIndex,
    build_junction_targets,
    collate_splice,
    densify_junction_targets,
)
from alphagenome_pytorch.losses import splice_junction_loss


def _spliser_rows():
    # + strand gene: donor 100 joins acceptors 200 (5 reads) and 300 (2 reads);
    # − strand gene: donor 900 joins acceptor 800 (4 reads).
    # Every junction is listed at both ends, as Spliser does.
    rows = [
        ("s1", "1", 100, "+", "{200: 5, 300: 2}"),
        ("s1", "1", 200, "+", "{100: 5}"),
        ("s1", "1", 300, "+", "{100: 2}"),
        ("s1", "1", 900, "-", "{800: 4}"),
        ("s1", "1", 800, "-", "{900: 4}"),
        ("s2", "1", 100, "+", "{200: 1, 300: 0}"),
        ("s2", "1", 200, "+", "{100: 1}"),
    ]
    return pd.DataFrame(rows, columns=["Sample", "Region", "Site", "Strand", "Partners"])


def test_extract_junctions_once_per_junction():
    jx = extract_junctions(_spliser_rows())
    got = sorted(zip(jx["Sample"], jx["Donor"], jx["Acceptor"], jx["Strand"], jx["Count"]))
    assert got == [
        ("s1", 100, 200, "+", 5),
        ("s1", 100, 300, "+", 2),
        ("s1", 900, 800, "-", 4),
        ("s2", 100, 200, "+", 1),  # zero-count partner dropped
    ]


def test_adjust_junction_positions_matches_site_shift():
    jx = adjust_junction_positions(extract_junctions(_spliser_rows()))
    plus = jx[jx["Strand"] == "+"]
    minus = jx[jx["Strand"] == "-"]
    assert set(plus["Donor"]) == {99}                  # + donor shifted
    assert set(plus["Acceptor"]) == {200, 300}         # + acceptor unchanged
    assert list(minus["Donor"]) == [900]               # − donor unchanged
    assert list(minus["Acceptor"]) == [799]            # − acceptor shifted


@pytest.fixture
def junction_parquet(tmp_path):
    df = _spliser_rows()
    df["Gene"] = "G"
    path = tmp_path / "Human.Brain.1.combined.tsv"
    df.to_csv(path, sep="\t", index=False)
    out = tmp_path / "junctions.parquet"
    convert_spliser_to_junction_parquet(
        [path], out, condition_labels={"Brain_0": 0, "Brain_1": 1}
    )
    return out


def test_converter_output(junction_parquet):
    df = pd.read_parquet(junction_parquet)
    assert list(df.columns) == ["Chromosome", "Donor", "Acceptor", "Strand",
                                "Condition", "Sample", "Count"]
    assert set(df["Condition"]) == {1}
    meta = json.loads(junction_parquet.with_suffix(".json").read_text())
    assert meta["condition_labels"] == {"Brain_0": 0, "Brain_1": 1}


def test_index_query_and_targets(junction_parquet):
    index = SpliceJunctionIndex(junction_parquet)
    assert index.n_conditions == 2

    donor, acceptor, plus, cond, count = index.query("1", 0, 1000)
    # Samples are summed per condition: 99→200 has 5 + 1 reads.
    got = sorted(zip(donor, acceptor, plus, cond, count))
    assert got == [(99, 200, True, 1, 6.0), (99, 300, True, 1, 2.0), (900, 799, False, 1, 4.0)]
    # Junctions must lie entirely inside the window.
    d, *_ = index.query("1", 150, 1000)
    assert sorted(d) == [900]
    assert len(index.query("2", 0, 1000)[0]) == 0

    win_start = 50
    positions, jx_index, jx_counts = build_junction_targets(
        donor - win_start, acceptor - win_start, plus, cond, count,
        n_conditions=index.n_conditions, max_sites=4,
    )
    assert positions.shape == (4, 4)
    assert positions[0].tolist() == [49, -1, -1, -1]       # donor+
    assert positions[1].tolist() == [150, 250, -1, -1]     # acceptor+
    assert positions[2].tolist() == [850, -1, -1, -1]      # donor−
    assert positions[3].tolist() == [749, -1, -1, -1]      # acceptor−

    dense = densify_junction_targets(
        torch.from_numpy(jx_index)[None], torch.from_numpy(jx_counts)[None],
        num_sites=4, num_tracks=2 * index.n_conditions,
    )
    assert dense.shape == (1, 4, 4, 4)
    assert dense[0, 0, 0, 1] == 6.0        # + strand, condition 1 → track 1
    assert dense[0, 0, 1, 1] == 2.0
    assert dense[0, 0, 0, 3] == 4.0        # − strand, condition 1 → track 2 + 1
    assert dense.sum() == 12.0


def test_build_targets_keeps_top_sites():
    donor = np.array([10, 10, 20])
    acceptor = np.array([30, 40, 40])
    plus = np.ones(3, dtype=bool)
    cond = np.zeros(3, dtype=np.int32)
    count = np.array([1.0, 1.0, 9.0], dtype=np.float32)
    positions, jx_index, jx_counts = build_junction_targets(
        donor, acceptor, plus, cond, count, n_conditions=1, max_sites=1
    )
    # Donor 20 (9 reads) beats donor 10 (2 reads); acceptor 40 (10) beats 30 (1).
    assert positions[0].tolist() == [20]
    assert positions[1].tolist() == [40]
    assert jx_index.tolist() == [[0, 0, 0]]
    assert jx_counts.tolist() == [9.0]


def test_collate_pads_junctions():
    def item(n):
        return {
            "sequence": torch.zeros(8, 4), "organism_index": torch.tensor(0),
            "classification_labels": torch.zeros(8, dtype=torch.long),
            "loss_mask": torch.ones(8, dtype=torch.bool),
            "window_start": torch.tensor(0), "chrom_idx": torch.tensor(0),
            "splice_site_positions": torch.full((4, 2), -1),
            "junction_index": torch.zeros((n, 3), dtype=torch.long),
            "junction_counts": torch.ones(n),
        }
    batch = collate_splice([item(3), item(0)])
    assert batch["junction_index"].shape == (2, 3, 3)
    assert (batch["junction_index"][1] == -1).all()
    assert batch["junction_counts"][1].sum() == 0


def _random_target(shape, mask):
    return torch.where(mask, torch.randint(0, 8, shape).float(), torch.zeros(shape))


def test_junction_loss_prefers_true_counts_and_ignores_masked():
    torch.manual_seed(0)
    shape = (2, 5, 5, 3)
    mask = torch.rand(shape) > 0.3
    target = _random_target(shape, mask)
    perfect = splice_junction_loss(y_true=target, y_pred=target + 1e-3, mask=mask)["loss"]
    wrong = splice_junction_loss(
        y_true=target, y_pred=torch.rand(shape) * 8, mask=mask
    )["loss"]
    assert perfect < wrong

    noisy = (target + 1e-3).clone()
    noisy[~mask] = 1e3
    masked = splice_junction_loss(y_true=target, y_pred=noisy, mask=mask)["loss"]
    torch.testing.assert_close(masked, perfect)


def test_head_factory_pretrained_init_and_backward():
    n_pre, h = 3, 768
    pretrained = create_splice_junction_finetuning_head(n_conditions=n_pre, num_organisms=2)
    state = pretrained.state_dict()

    head = create_splice_junction_finetuning_head(
        n_conditions=2, num_organisms=3, pretrained_state=state,
        organism_init={0: 0, 1: 1, 2: 1}, pretrained_num_tissues=(3, 2),
    )
    torch.testing.assert_close(head.conv.weight[2], state["conv.weight"][1])
    expected = state["rope_params.pos_donor"][1][:, :2].mean(dim=1)       # (2, H)
    torch.testing.assert_close(head.rope_params["pos_donor"][2, :, 0], expected)
    torch.testing.assert_close(head.rope_params["pos_donor"][2, :, 1], expected)

    seq_len = 32
    emb = torch.randn(1, 1536, seq_len)
    positions = torch.tensor([[[1, 4, -1], [10, 12, -1], [20, -1, -1], [15, -1, -1]]])
    out = head(emb, torch.tensor([2]), splice_site_positions=positions)
    assert out["pred_counts"].shape == (1, 3, 3, 4)
    target = torch.zeros_like(out["pred_counts"])
    target[0, 0, 1, 0] = 3.0
    loss = splice_junction_loss(
        y_true=target, y_pred=out["pred_counts"], mask=out["splice_junction_mask"]
    )["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert head.rope_params["pos_donor"].grad is not None


class _StubTrunk(torch.nn.Module):
    """Stands in for AlphaGenome: returns learnable 1 bp embeddings."""

    def __init__(self, seq_len):
        super().__init__()
        from alphagenome_pytorch.extensions.finetuning.heads import (
            create_splice_classification_finetuning_head,
        )
        self.emb = torch.nn.Parameter(torch.randn(1, 1536, seq_len) * 0.1)
        self.splice_sites_classification_head = create_splice_classification_finetuning_head(
            num_organisms=1
        )

    def forward(self, seq, org_idx, **kwargs):
        return {"embeddings_1bp": self.emb.expand(seq.shape[0], -1, -1)}


def test_training_loop_with_junction_head():
    from alphagenome_pytorch.extensions.finetuning.splice_training import (
        train_epoch_splice,
        validate_splice,
    )

    torch.manual_seed(0)
    seq_len, n_cond = 32, 2
    positions, jx_index, jx_counts = build_junction_targets(
        np.array([2, 2, 25]), np.array([10, 14, 18]), np.array([True, True, False]),
        np.array([0, 1, 0], dtype=np.int32), np.array([4.0, 1.0, 3.0], dtype=np.float32),
        n_conditions=n_cond, max_sites=4,
    )
    item = {
        "sequence": torch.zeros(seq_len, 4), "organism_index": torch.tensor(0),
        "classification_labels": torch.full((seq_len,), 4, dtype=torch.long),
        "loss_mask": torch.ones(seq_len, dtype=torch.bool),
        "window_start": torch.tensor(0), "chrom_idx": torch.tensor(0),
        "splice_site_positions": torch.from_numpy(positions),
        "junction_index": torch.from_numpy(jx_index),
        "junction_counts": torch.from_numpy(jx_counts),
    }
    loader = [collate_splice([item, item])] * 4

    model = _StubTrunk(seq_len)
    heads = {0: create_splice_junction_finetuning_head(n_conditions=n_cond)}
    params = list(model.parameters()) + list(heads[0].parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)
    device = torch.device("cpu")

    before = validate_splice(model, None, loader, device, junction_head=heads, junction_weight=1.0)
    assert before.junction_loss > 0
    for epoch in range(1, 4):
        train_epoch_splice(model, None, loader, optimizer, None, device,
                           junction_head=heads, junction_weight=1.0, log_every=0, epoch=epoch)
    after = validate_splice(model, None, loader, device, junction_head=heads, junction_weight=1.0)
    assert after.junction_loss < before.junction_loss
    assert "val_junction_loss" in after.species_metrics[0]

    # junction_weight=0 leaves the junction head out entirely.
    off = validate_splice(model, None, loader, device, junction_head=heads, junction_weight=0.0)
    assert off.junction_loss == 0.0


def test_junction_tissue_init_matches_by_name(tmp_path):
    from finetune_splice import junction_tissue_init

    meta = pd.DataFrame({
        "idx": [0, 1, 2, 0],
        "biosample_name": ["Testis", "liver", "right lobe of liver", "testis"],
        "output_type": ["OutputType.SPLICE_JUNCTIONS"] * 3 + ["OutputType.ATAC"],
    })
    csv = tmp_path / "meta.csv"
    meta.to_csv(csv, index=False)
    got = junction_tissue_init({"Testis_1": 0, "Liver_2": 1, "Ovary_1": 2}, {0: str(csv)})
    assert got == {0: {0: [0], 1: [1, 2], 2: []}}

    state = create_splice_junction_finetuning_head(n_conditions=3, num_organisms=1).state_dict()
    head = create_splice_junction_finetuning_head(
        n_conditions=3, num_organisms=1, pretrained_state=state,
        pretrained_num_tissues=(3,), tissue_init=got,
    )
    src = state["rope_params.pos_acceptor"][0]                             # (2, 3, H)
    rope = head.rope_params["pos_acceptor"][0]
    torch.testing.assert_close(rope[:, 0], src[:, 0])                      # testis
    torch.testing.assert_close(rope[:, 1], src[:, 1:3].mean(dim=1))        # liver
    torch.testing.assert_close(rope[:, 2], src.mean(dim=1))                # unmatched → mean
