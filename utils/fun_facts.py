# location: /utils/fun_facts.py

import random

FACTS = [
    "GANs were introduced by Ian Goodfellow and collaborators in 2014.",
    "A GAN trains two networks together: a generator and a discriminator.",
    "StyleGAN popularized style-based controls for high-quality image synthesis.",
    "CycleGAN can learn image-to-image translation without paired examples.",
    "GAN training can be unstable because the two networks are constantly adapting.",
    "Conditional GANs guide generation with labels or other structured inputs.",
    "Latent vectors are compact coordinates that a generator maps into images.",
]


def get_fun_fact() -> str:
    return random.choice(FACTS)
