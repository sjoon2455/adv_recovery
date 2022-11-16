# Recovering Adversarial Image in Latent Vector Space
This project provides three different ways to recover an adversarial image.
- Attack again to recover (_attack_to_recover.py_): Given an attacked image, apply an attack once more to send it to where it belonged to.
- Genetic Algorithm in raw image space (_ga.py_): Given an attacked image, apply a genetic algorithm guided by Surprise Adequacy.
- Local search in latent vector space (_local_search.py_): Given an attacked image, apply a local search guided by Surprise Adequacy in a latent vector space encoded by VAE encoder.

### Experiment
A simple experiment can be conducted by executing _experiment.py_. All the pretrained models are available in _/model_.
