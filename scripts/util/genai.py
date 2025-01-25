import os
import json
import litellm
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class GenerateLLMResponse:

    def __init__(self, model_id=None):
        """
        Initialize class with a model ID.
        :param model_id: str, model id/name (optional). Defaults to environment variable MODEL_ID.
        """
        self.model_id = model_id or os.getenv("MODEL_ID")
        if not self.model_id: raise ValueError("Model ID required.")


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


