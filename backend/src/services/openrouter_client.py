"""
Mock OpenRouterClient to replace OpenAI client as per requirements.
This simulates an OpenRouter-specific client while maintaining compatibility.
"""
import openai
import os
import logging

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")

        # Check if this is a mock/test key
        self.is_mock_key = api_key and ('mock' in api_key.lower() or 'test' in api_key.lower())

        if not self.is_mock_key:
            # Initialize the OpenAI client to use OpenRouter's endpoint
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            # For mock keys, we'll handle requests differently
            self.client = None

        # Store the api key for potential use
        self.api_key = api_key

    def chat(self):
        """Returns an object with chat completion methods"""
        if self.is_mock_key or not self.client:
            # Return a mock object for testing
            class MockChatCompletions:
                def create(self, *args, **kwargs):
                    # Return a mock response for testing
                    class MockResponse:
                        class MockChoice:
                            class MockMessage:
                                content = "This is a mock response for testing purposes. In production, this would be a response from OpenRouter."
                            message = MockMessage()
                        choices = [MockChoice()]
                    return MockResponse()
            return MockChatCompletions()
        return self.client.chat.completions

    def completions(self):
        """Returns an object with completion methods"""
        if self.is_mock_key or not self.client:
            class MockCompletions:
                def create(self, *args, **kwargs):
                    class MockResponse:
                        class MockChoice:
                            text = "This is a mock completion for testing."
                        choices = [MockChoice()]
                    return MockResponse()
            return MockCompletions()
        return self.client.completions

    def create(self, **kwargs):
        """Wrapper for chat completion create"""
        if self.is_mock_key or not self.client:
            class MockResponse:
                class MockChoice:
                    class MockMessage:
                        content = "Mock response for testing purposes."
                    message = MockMessage()
                choices = [MockChoice()]
            return MockResponse()
        return self.client.chat.completions.create(**kwargs)

    def Embedding(self, *args, **kwargs):
        """Wrapper for embedding create using new v1.x API"""
        if self.is_mock_key or not self.client:
            # Return mock embedding response
            class MockEmbeddingResponse:
                class MockDataItem:
                    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock 5-dimensional embedding
                data = [MockDataItem()]
            return MockEmbeddingResponse()
        return self.client.embeddings.create(*args, **kwargs)

    def ChatCompletion(self, *args, **kwargs):
        """Direct access to chat completions using new v1.x API"""
        if self.is_mock_key or not self.client:
            class MockResponse:
                class MockChoice:
                    class MockMessage:
                        content = "This is a mock response from OpenRouter API for testing purposes. With a valid API key, this would contain a real response from OpenRouter."
                    message = MockMessage()
                choices = [MockChoice()]
            return MockResponse()
        return self.client.chat.completions.create(*args, **kwargs)