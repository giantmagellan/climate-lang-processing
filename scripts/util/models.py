import os


class ModelIDManager:
    """
    A helper class to manage and retrieve model IDs for different hosts.
    """
    def __init__(self):
        """
        Initialize with a mapping of host types to environment variable names.
        """
        self.host_to_env_var = {
            "litellm": "LITELLM_MODEL_ID",
            "bedrock": "BEDROCK_MODEL_ID"
        }

    def get_model_id(self, host: str) -> str:
        """
        Retrieve the model ID for the specified host.
        :param host: str, the name of the host (e.g., 'litellm', 'bedrock').
        :return: str, the model ID.
        """
        env_var = self.host_to_env_var.get(host.lower())
        if not env_var:
            raise ValueError(f"Unsupported host '{host}'. Supported hosts: {list(self.host_to_env_var.keys())}.")

        model_id = os.getenv(env_var)
        if not model_id:
            raise ValueError(f"Model ID for '{host}' is not set in the environment variable '{env_var}'.")
        return model_id
