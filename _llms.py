# taken from my dissertaion, an older implementation
import os
import yaml
import base64
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union, Iterable

class LLMHelper:
    _instance = None  # Class-level attribute to hold the singleton instance

    """
    A helper library for making LLM queries using the openai library,
    supporting various providers and functionalities like toggling thinking,
    attaching images, streaming, and storing chat history.
    """

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMHelper, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = 'llms.yml', env_path: str = '.env'):
        """
        Initializes the LLMHelper.

        Args:
            config_path (str): Path to the YAML configuration file.
            env_path (str): Path to the .env file with API keys.
        """
        if not hasattr(self, '_initialized') or not self._initialized:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            load_dotenv(dotenv_path=env_path)
            self.history_dir = "chat_history"
            os.makedirs(self.history_dir, exist_ok=True)

            self._initialized = True  # Mark the instance as initialized

    def _get_client(self, provider: str) -> OpenAI:
        """
        Initializes the OpenAI client for a given provider.

        Args:
            provider (str): The name of the LLM provider.

        Returns:
            OpenAI: An instance of the OpenAI client.
        """
        if provider not in self.config:
            raise ValueError(f"Provider '{provider}' not found in the configuration.")

        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        if not api_key and provider != "ollama":
            raise ValueError(f"API key for provider '{provider}' not found in .env file.")

        base_url = self.config[provider].get('url')
        return OpenAI(api_key=api_key, base_url=base_url)

    def _encode_image(self, image_path: str) -> str:
        """
        Encodes an image file to base64.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64 encoded image string.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_last_message(self, session_id: str = 'default') -> Optional[str]:
        """
        Retrieves the last message from the chat history for a given session.
        Args:
            session_id (str): The unique identifier for the chat session.

        Returns:
            Optional[str]: The last message from the chat history, or None if no history exists.
        """
        history = self._load_history(session_id)
        if history:
            last_message = history[-1]
            if last_message.get('role') == 'assistant':
                return last_message.get('content', '')
        return None

    # TODO: our history contains extra fields!
    def _load_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Loads the chat history for a given session.

        Args:
            session_id (str): The unique identifier for the chat session.

        Returns:
            List[Dict[str, str]]: The chat history.
        """
        history_file = os.path.join(self.history_dir, f"{session_id}.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return []

    def _save_history(self, session_id: str, history: List[Dict[str, str]]):
        """
        Saves the chat history for a given session.

        Args:
            session_id (str): The unique identifier for the chat session.
            history (List[Dict[str, str]]): The chat history to save.
        """
        history_file = os.path.join(self.history_dir, f"{session_id}.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def query(self,
              provider: str,
              model: str,
              prompt: str,
              session_id: str = 'default',
              image_paths: Optional[List[str]] = None,
              stream: bool = False,
              full_history: Optional[List[Dict[str, str]]] = None,
              return_stream_generator: bool = True) -> Union[str, None, Iterable[str]]:
        """
        Makes a query to the specified LLM.

        Args:
            provider (str): The LLM provider (e.g., 'openai').
            model (str): The model to use (e.g., 'gpt-4o-mini').
            prompt (str): The user's prompt.
            session_id (str): A unique identifier for the chat session to maintain history.
            image_paths (Optional[List[str]]): A list of paths to images to include in the prompt.
            stream (bool): Whether to stream the response.
            history (Optional[List[Dict[str, str]]]): Manually provided chat history.
            return_stream_generator (bool): Whether to return a streaming generator or print chunks and return full string.

        Returns:
            Union[str, None, Iterable[str]]: The LLM's response as a string, or a generator yielding response chunks if streaming is enabled.
        """
        if full_history:
            # Ensure the session_id is new and does not conflict with existing sessions
            history_file = os.path.join(self.history_dir, f"{session_id}.json")
            if os.path.exists(history_file):
                raise ValueError(f"Session ID '{session_id}' already exists. Please use a new session ID for custom history.")
            # Save the provided history to the new session ID
            self._save_history(session_id, full_history)
        else:
            # Load history automatically if not provided
            full_history = self._load_history(session_id)
            history_for_request = self._load_history_compatible(session_id)  # Ensure compatibility with chat completions requests

        client = self._get_client(provider)
        model_config = self.config.get(provider, {}).get('models', {}).get(model)

        if not model_config:
            raise ValueError(f"Model '{model}' not found for provider '{provider}' in the configuration.")

        content: List[Dict[str, str]] = [{"type": "text", "text": prompt}]

        if image_paths:
            if not model_config.get('vision'):
                raise ValueError(f"Model '{model}' does not support vision.")
            for image_path in image_paths:
                base64_image = self._encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

        messages = full_history + [{"role": "user", "content": content}]

        if model_config.get('thinking', False):
            print("Thinking...")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=model_config.get('t', 0.5)
            )

            if stream:
                if return_stream_generator:
                    def response_generator():
                        full_response = ""
                        for chunk in response:
                            content_chunk = chunk.choices[0].delta.content
                            if content_chunk:
                                print(content_chunk, end='', flush=True)
                                full_response += content_chunk
                                yield content_chunk
                        print()
                        full_history.append({"role": "user", "content": prompt})
                        full_history.append({
                            "role": "assistant", 
                            "content": full_response,
                            "model": model,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "image_paths": image_paths or []
                        })
                        self._save_history(session_id, full_history)

                    return response_generator()
                else:
                    full_response = ""
                    for chunk in response:
                        content_chunk = chunk.choices[0].delta.content
                        if content_chunk:
                            print(content_chunk, end='', flush=True)
                            full_response += content_chunk
                    print()
                    full_history.append({"role": "user", "content": prompt})
                    full_history.append({
                        "role": "assistant", 
                        "content": full_response,
                        "model": model,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "image_paths": image_paths or []
                    })
                    self._save_history(session_id, full_history)
                    return full_response
            else:
                assistant_response = response.choices[0].message.content
                full_history.append({"role": "user", "content": prompt})
                full_history.append({
                    "role": "assistant", 
                    "content": assistant_response,
                    "model": model,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "image_paths": image_paths or []
                })
                self._save_history(session_id, full_history)
                return assistant_response

        except Exception as e:
            return f"An error occurred: {e}"

    def extract_jsonl_from_response(self, response: str) -> List[Dict]:
        """
        Extracts a JSONL (JSON Lines) markdown code block from an LLM response and parses it.

        Args:
            response (str): The LLM response containing a JSONL markdown code block.

        Returns:
            List[Dict]: A list of dictionaries parsed from the JSONL code block.

        Raises:
            ValueError: If no valid JSONL code block is found or parsing fails.
        """
        lines = response.splitlines()
        start, end = None, None

        # Find the start and end of the JSONL block
        for i, line in enumerate(lines):
            if line.strip() == "```jsonl":
                start = i + 1
            elif line.strip() == "```" and start is not None:
                end = i
                break

        if start is None or end is None:
            raise ValueError("No JSONL markdown code block found in the response.")

        jsonl_block = lines[start:end]
        parsed_lines = []

        try:
            for line in jsonl_block:
                parsed_lines.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSONL: {e}")

        return parsed_lines
    
    def extract_python_from_response(self, response: str) -> str:
        """
        Extracts a Python code block from an LLM response.

        Args:
            response (str): The LLM response containing a Python code block.

        Returns:
            str: A string representing the Python code block.

        Raises:
            ValueError: If no valid Python code block is found.
        """
        lines = response.splitlines()
        start, end = None, None

        # Find the start and end of the Python code block
        for i, line in enumerate(lines):
            if line.strip() == "```python":
                start = i + 1
            elif line.strip() == "```" and start is not None:
                end = i
                break

        if start is None or end is None:
            raise ValueError("No Python markdown code block found in the response.")

        return "\n".join(lines[start:end])

    # at this point I decided I need a better design
    def _load_history_compatible(self, session_id: str) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        """
        Loads the chat history for a given session and ensures compatibility with chat completions requests.

        Args:
            session_id (str): The unique identifier for the chat session.

        Returns:
            List[Dict[str, Union[str, List[Dict[str, str]]]]]: The transformed chat history compatible with chat completions requests.
        """
        history = self._load_history(session_id)
        compatible_history = []

        for entry in history:
            if 'role' in entry and 'content' in entry:
                compatible_entry = {
                    "role": entry["role"],
                    "content": entry["content"]
                }
                compatible_history.append(compatible_entry)

        return compatible_history

if __name__ == '__main__':
    # Example Usage
    helper = LLMHelper()

    # --- Text-only query ---
    # print("--- Text-only Query ---")
    # response_text = helper.query(
    #     provider='openai',
    #     model='gpt-4o-mini',
    #     prompt='Hello, who are you?',
    #     session_id='my_text_session'
    # )
    # if response_text:
    #     print(f"Response: {response_text}")

    # --- Text-only query with streaming ---
    # print("\n--- Text-only Query with Streaming ---")
    # helper.query(
    #     provider='google',
    #     model='gemini-2.5-flash-lite-preview-06-17',
    #     prompt='Tell me a short story.',
    #     session_id='gemini_streaming_session',
    #     stream=True
    # )

    # --- Vision query with multiple images in one round ---
    try:
        from PIL import Image

        # Create five dummy images of different colors
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        image_paths = []

        for color in colors:
            img = Image.new('RGB', (60, 30), color=color)
            image_path = f'dummy_image_{color}.png'
            img.save(image_path)
            image_paths.append(image_path)

        print("\n--- Vision Query with Multiple Images in One Round ---")
        response_vision = helper.query(
            provider='openai',
            model='gpt-4o-mini',
            prompt='What are the colors of the images in the order they are attached?',
            session_id='my_vision_session_all',
            image_paths=image_paths
        )
        if response_vision:
            print(f"Response: {response_vision}")

    except ImportError:
        print("\nSkipping vision query example because Pillow is not installed.")
        print("Install it with: pip install Pillow")
    except Exception as e:
        print(f"\nAn error occurred during the vision query example: {e}")

    # --- Query with a provider that has "thinking" enabled ---
    # Note: This requires a valid API key for the specified provider in your .env
    # try:
    #     print("\n--- Query with 'thinking' enabled (using DeepSeek as an example) ---")
    #     # This will print "Thinking..." before making the API call.
    #     # Ensure you have a DEEPSEEK_API_KEY in your .env file.
    #     response_thinking = helper.query(
    #         provider='deepseek',
    #         model='deepseek-reasoner',
    #         prompt='What is the speed of light?',
    #         session_id='my_thinking_session',
    #         stream=True
    #     )
    #     if response_thinking:
    #         print(f"Response: {response_thinking}")
    # except ValueError as e:
    #     print(f"Skipping 'thinking' example: {e}")