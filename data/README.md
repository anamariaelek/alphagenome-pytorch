
## stage_age_table.csv

Aligned pipeline timepoint -> absolute age (days post conception) for the seven
evo-devo species; 82 species x timepoint rows over 2,450 samples.

Regenerate the table and its figure with:
    python stage_age_alignment.py            # reads ../../spliser/data/samples*.txt
    python stage_age_alignment.py --samples-dir DIR --out-dir DIR

stage_age_alignment.py and stage_age_alignment.png live here rather than in
code/scripts/ so the figure, its generating code and its input table stay together;
the figure has no dependency on the AlphaGenome prediction store, which is why it is
not produced by devas_report.py.

Run it in the pipeline env (the login shell's python has no numpy):
    source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate alphagenome_pytorch_genomicsxai
Verified 7 Sep 2026: rerunning from spliser/data/samples*.txt reproduces
stage_age_table.csv byte-identically.
