import os
import json
import litellm
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from models import ModelIDManager


class LLMResponseGenerator:

    def __init__(self, host=str):
        """
        Initialize class with a model ID.
        :param model_id: str, model id/name (optional). Defaults to environment variable MODEL_ID.
        """
        self.host = host
        self.model_id_manager = ModelIDManager()
        self.model_id = self.model_id_manager.get_model_id(host)

    def generate_groq_response(self, prompt: str) -> str:
        """
        Make request to LLM and generate a response.
        :param prompt: str, instruction text.
        :param model_id: str, model id/name
        :return: str, generate LLM response
        """
        response = litellm.completion(
            model = self.model_id,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.7,
            api_key = os.environ['GROQ_API_KEY'],
            drop_params=True
        )
        return response.choices[0].message.content

    @staticmethod
    def _get_bedrock_client():
        """
        Create and return a Bedrock client with custom configuration.
        :return: boto3 client for Bedrock
        """
        botocore_config: Config = Config(
            connect_timeout=15, 
            read_timeout=60, 
            retries={'max_attempts': 3}
            )
        return boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=os.environ["AWS_BEDROCK_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_BEDROCK_SECRET_ACCESS_KEY"],
            config=botocore_config
        )

    def generate_bedrock_response(self, prompt: str) -> str:
        """
        Make request to LLM and generate a response.
        :param prompt: str, instruction text.
        :param model_id: str, model id/name
        :return: str, generate LLM response
        """
        client = self._get_bedrock_client()

        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.9,
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        try:
            # Invoke the model with the request.
            response = client.invoke_model(
                modelId=self.model_id, 
                body=request
                )
            # Decode the response body.
            model_response = json.loads(response["body"].read())
            return model_response["generation"]

        except (ClientError, Exception) as e:
            print(f"Can't invoke '{self.model_id}'. Reason: {e}")


