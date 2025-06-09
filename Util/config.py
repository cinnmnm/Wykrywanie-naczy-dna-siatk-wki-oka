import os
import yaml

class Config:
    @staticmethod
    def load(root_path="config.yaml", local_path=None):
        """
        Load a YAML config file. If local_path is provided and exists, merge it over the root config.
        Local config values override root config values.
        """
        with open(root_path, "r") as f:
            config = yaml.safe_load(f)
        if local_path and os.path.exists(local_path):
            with open(local_path, "r") as f:
                local_config = yaml.safe_load(f)
            config.update(local_config)
        return config