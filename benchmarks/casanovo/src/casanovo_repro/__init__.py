"""Faithful retrain of Casanovo (Yilmaz et al. 2024, Nat Commun) on Polaris.

Wraps upstream casanovo without forking: a paper-faithful YAML overlay is
deep-merged with each per-dataset overlay (massive_kb / consensus), then
casanovo.config.Config is constructed and casanovo.denovo.ModelRunner runs the
training. The wrapper exists only to (a) split the consensus train MGF into
train_main + val, (b) trigger casanovo's depthcharge-based Lance build in a
preprocess-only invocation, and (c) inject a Lightning WandbLogger into the
Trainer that ModelRunner.initialize_trainer constructs.
"""
