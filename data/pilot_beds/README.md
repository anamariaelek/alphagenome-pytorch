# Pilot BED subsets

Fixed-seed random subsamples of each species' `results_orthologs_132kb_protein_coding/<species>/fold0/{train,valid,test}.bed`
(2000 train / 500 valid / 500 test windows per species, `random.Random(1950).sample(...)`,
re-sorted by genomic position afterward for readability). Not a canonical
data split — exist only to make `configs/hqmrboc_phase1_pilot.yaml` fast
enough to iterate on (full fold0 splits are ~18-19k train windows per
species, which at ~26s/optimizer-step made even one epoch take hours).

`train.bed`/`valid.bed` are read by `finetune_splice.py` for the pilot
training run. `test.bed` isn't read by `finetune_splice.py` at all (it only
uses `train_bed`/`val_bed`) — it exists for a later `evaluate_splice.py
--data-config <json>` held-out eval pass, whose species-spec JSON schema
(`genome`/`annotation_parquet`/`usage_parquet`/`train_bed`/`val_bed`/
`test_bed`/`organism_index`) matches these directories' layout.

Currently covers: human, mouse, macaque.

Regenerate with a different size/species set:

```python
import random, os

SPECIES = {
    "human": "/path/to/results_orthologs_132kb_protein_coding/human/fold0",
    ...
}
N_TRAIN, N_VAL, N_TEST, SEED = 2000, 500, 500, 1950

for sp, d in SPECIES.items():
    rng = random.Random(SEED)
    for split, n in (("train", N_TRAIN), ("valid", N_VAL), ("test", N_TEST)):
        with open(f"{d}/{split}.bed") as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        if len(lines) > n:
            lines = rng.sample(lines, n)
        lines.sort(key=lambda l: (l.split("\t")[0], int(l.split("\t")[1])))
        os.makedirs(f"data/pilot_beds/{sp}", exist_ok=True)
        with open(f"data/pilot_beds/{sp}/{split}.bed", "w") as f:
            f.writelines(lines)
```
