# CLAM-bcda (fork) - Repo Guidelines

Purpose:
- This fork is the canonical place for structural model changes to CLAM.
- Use it to track, tag, and share model architecture changes (forward(), pooling, attention modules, loss functions).

What belongs here:
- Modifications to model implementations (e.g., `models/model_clam.py`).
- Re-architecting attention/pooling heads or adding new modules.
- Unit tests for model components.
- Tags for model versions (e.g., `v1_conch768_patch`).

What does NOT belong here:
- Experiment glue code such as dataset paths, server-specific sbatch scripts, or run orchestration.
- Large data, feature files, or training results (keep those in the lab project folder).

Tagging recommendation:
- After structural changes that affect model checkpoint shapes, tag the commit with a semantic name and note the important parameters (embed_dim, model_size) in the tag or a short release note.

Example:
```bash
# Tagging a model version
git tag -a v1_conch768_patch -m "CLAM_SB with CONCH 768-dim embeddings" 
git push origin v1_conch768_patch
```

Contact:
- Keep this fork minimal and well-documented; for experiment scripts and dataset glue, use your personal `bcda` repo.
