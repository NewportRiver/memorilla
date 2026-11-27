# LITE-008W-R3 Work-Unit Contract

Each direction is an independently identified work unit. Its identity binds manifest hash, partition, entity, row ID, target, exact evidence and query, candidate strings and token IDs, factual direction, encoder identity, decoder identity, and teacher configuration.

Each update atomically checkpoints latent vector, optimizer state, trajectory, completed count, identity, and configuration. Completion atomically writes the teacher vector and metadata, then reload-verifies both identity and update count. Incomplete checkpoints cannot be promoted to teacher artifacts.

Implementation: `experiments/lite008w/work_units.py`, launcher wrapper `run_work_unit.py`. Soft scheduling deadline is 240 seconds; hard limit remains 300 seconds.
