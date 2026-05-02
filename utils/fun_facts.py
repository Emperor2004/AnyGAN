# location: /utils/fun_facts.py

import random

FACTS = [
    "Stable Diffusion generates images by gradually denoising latent representations.",
    "A fixed seed can reproduce the same image when the model and settings are unchanged.",
    "Classifier-free guidance helps steer generated images toward the prompt.",
    "Negative prompts describe visual traits the model should avoid.",
    "Diffusion models usually trade speed for quality through the number of inference steps.",
    "Latent diffusion works in a compressed image space instead of raw pixels.",
    "Prompt wording, seed, guidance, and model checkpoint can all noticeably change the output.",
]


def get_fun_fact() -> str:
    return random.choice(FACTS)
