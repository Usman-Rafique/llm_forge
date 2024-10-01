from .gpt import GPTModel


class ModelFactory:

    @staticmethod
    def create_model(model_type, config):
        if model_type == 'gpt':
            return GPTModel(**config)
        # Other model types will be added here
        else:
            raise ValueError(f"Unknown model type: {model_type}")
