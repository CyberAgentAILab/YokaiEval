import os
from openai import AzureOpenAI, BadRequestError
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_MODEL = os.getenv("AZURE_OPENAI_API_MODEL", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")


class GPT:
    def __init__(
        self,
        model: str = AZURE_OPENAI_API_MODEL,
        api_version: str = AZURE_OPENAI_API_VERSION,
    ):
        # you can check the newest price in the Azure Openai official website: https://azure.microsoft.com/ja-jp/pricing/details/cognitive-services/openai-service/
        # [price per token for input tokens (prompt_tokens), price per token for output tokens (completion_tokens)]
        self.price = [
            0.0000000165,
            0.000000066,
        ]  # $0.165 per 1M tokens, $0.66 per 1M tokens
        self.model = AZURE_OPENAI_API_MODEL
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

        self.total_cost = 0.0

    def _update(self, response):
        current_cost = (
            response.usage.prompt_tokens * self.price[0]
            + response.usage.completion_tokens * self.price[1]
        )
        self.total_cost += current_cost
        print(
            f"Current Tokens: {response.usage.prompt_tokens+response.usage.completion_tokens:d}, Current cost: {current_cost:.4f} $, Total cost: {self.total_cost:.4f} $",
        )

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 200):
        """Input Format:
        messages: list of dict
            [{"role": "system", "content": "system message"}, {"role": "user", "content": "user message"}]
        temperature: float
            Lower values for temperature result in more consistent outputs (e.g. 0.2), while higher values generate more diverse and creative results (e.g. 1.0).
        max_tokens: int
            The maximum number of tokens to generate.
        """
        generated_text = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._update(generated_text)
        generated_text = generated_text.choices[0].message.content
        return generated_text

    def send_user_prompt(self, prompt: str) -> str:
        """Send a user message to GPT and get the response

        Args:
            gpt (GPT): GPT object
            prompt (str): prompt content

        Returns:
            str: Response from GPT
        """
        messages = [
            {"role": "user", "content": prompt},
        ]
        try:
            response_from_gpt = self.chat(messages)
        except BadRequestError as e:
            print(f"Error: {e}")
            return ""
        return response_from_gpt
