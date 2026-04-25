# location: /models/base_model.py

class BaseGAN:
    def generate(self, params):
        raise NotImplementedError("Generate method must be implemented")