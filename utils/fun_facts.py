# location: /utils/fun_facts.py

import random

facts = [
    "GANs were invented by Ian Goodfellow in 2014.",
    "GANs consist of a generator and a discriminator.",
    "StyleGAN can create hyper-realistic human faces.",
    "GANs are used in deepfakes and art generation.",
    "Training GANs is notoriously unstable."
]

def get_fun_fact():
    return random.choice(facts)