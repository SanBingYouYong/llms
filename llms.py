import base64
import datetime
import json
import os
import shutil
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import openai
import yaml
from dotenv import load_dotenv

# --- Constants and Configuration ---
HISTORY_DIR = Path("./chat_history")
CONFIG_FILE = Path("./llms.yml")


class LLMHelper:
    """A reusable helper library for interacting with Language and Vision Models."""

    def __init__(self):
        """Initializes the helper, loading configurations and API keys."""
        load_dotenv()
        self._load_config()
        self._clients = {}

    def _load_config(self):
        """Loads provider configurations from llms.yml."""
        if not CONFIG_FILE.is_file():
            raise FileNotFoundError(
                f"Configuration file not found at: {CONFIG_FILE.resolve()}"
            )
        with open(CONFIG_FILE, "r") as f:
            self.config = yaml.safe_load(f)

    def _get_client(self, provider: str) -> openai.OpenAI:
        """
        Retrieves or creates an API client for the specified provider.

        Args:
            provider: The name of the API provider.

        Returns:
            An instance of the OpenAI client.
        """
        if provider not in self._clients:
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                raise ValueError(f"API key for {provider} not found in .env file.")
            provider_config = self.config.get(provider, {})
            self._clients[provider] = openai.OpenAI(
                base_url=provider_config.get("url"), api_key=api_key
            )
        return self._clients[provider]

    def _verify_model_config(
        self, provider: str, model: str, image_paths: List[str]
    ):
        """
        Verifies the provider and model configurations.

        Args:
            provider: The name of the API provider.
            model: The alias of the model.
            image_paths: A list of absolute paths to image files.

        Raises:
            ValueError: If the provider, model, or configuration is invalid.
        """
        if provider not in self.config:
            raise ValueError(f"Provider '{provider}' not found in {CONFIG_FILE}")
        if model not in self.config[provider]["models"]:
            raise ValueError(
                f"Model '{model}' not found for provider '{provider}' in {CONFIG_FILE}"
            )

        model_config = self.config[provider]["models"][model]
        if image_paths and not model_config.get("vision", False):
            raise ValueError(f"Model '{model}' does not support vision.")

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        Encodes an image file to a base64 string.

        Args:
            image_path: The absolute path to the image file.

        Returns:
            The base64 encoded image string.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at: {image_path}")

    def _trim_and_convert_image_paths_to_urls(self, history_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts image paths in history messages to image URLs with base64 encoded images. Also leaves only role and content keys for API compatibility.

        Args:
            history_messages: A list of previous chat messages.

        Returns:
            A list of messages with image paths converted to image URLs.
        """
        converted_messages = []
        for msg in history_messages:
            if "role" in msg and "content" in msg:
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    converted_content = []
                    for item in msg["content"]:
                        if item.get("type") == "image_path" and "path" in item:
                            base64_image = self._encode_image(item["path"])
                            converted_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                }
                            )
                        else:
                            converted_content.append(item)
                    converted_messages.append({"role": "user", "content": converted_content})
                else:
                    converted_messages.append({"role": msg["role"], "content": msg["content"]})
        return converted_messages

    def _prepare_messages(
        self,
        history_messages: List[Dict[str, Any]],
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list for the API request.

        Args:
            history_messages: A list of previous chat messages.
            user_prompt: The current user query.
            image_paths: A list of absolute paths to image files.

        Returns:
            The formatted list of messages for the API.
        """
        messages = self._trim_and_convert_image_paths_to_urls(history_messages)
        content = [{"type": "text", "text": user_prompt}]
        if image_paths:
            for path in image_paths:
                base64_image = self._encode_image(path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
        messages.append({"role": "user", "content": content})
        return messages

    def _save_history(
        self,
        save_path: Optional[str],
        history_to_save: List[Dict[str, Any]],
    ):
        """
        Saves the chat history to a JSON file.

        Args:
            save_path: The path specification for saving.
            history_to_save: The history list to be saved.
        """
        if save_path is None:
            return

        HISTORY_DIR.mkdir(exist_ok=True)
        if save_path == "":
            session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            filepath = HISTORY_DIR / f"{session_id}.json"
        else:
            filepath = HISTORY_DIR / f"{save_path}.json"

        with open(filepath, "w") as f:
            json.dump(history_to_save, f, indent=2)

    def query(
        self,
        provider: str,
        model: str,
        user_prompt: str,
        history_messages: List[Dict[str, Any]] = [],
        image_paths: Optional[List[str]] = None,
        streamed: bool = False,
        return_stream_generator: bool = False,
        print_to_console: bool = False,
        save_path: Optional[str] = "",
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Performs a query to the specified LLM/VLM.

        Args:
            provider: The name of the API provider.
            model: The alias of the model.
            user_prompt: The current user query.
            history_messages: A list of previous chat messages.
            image_paths: A list of absolute paths to image files.
            streamed: Whether to use stream mode.
            return_stream_generator: If streamed, return a generator.
            print_to_console: Whether to print the response to the console.
            save_path: Path to save chat history.
            **kwargs: Additional parameters for the API call (e.g., temperature).

        Returns:
            The LLM's response as a string or a generator.
        """
        image_paths = image_paths or []
        self._verify_model_config(provider, model, image_paths)

        model_config = self.config[provider]["models"][model]
        api_model_name = model_config["model"]
        final_kwargs = model_config.copy()
        final_kwargs.update(kwargs)
        # Clean up config keys that are not part of the API call and are just model capability indicators
        for key in ["model", "vision", "thinking"]:
            final_kwargs.pop(key, None)

        messages = self._prepare_messages(history_messages, user_prompt, image_paths)
        client = self._get_client(provider)

        request_time = datetime.datetime.now().isoformat()

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if streamed:
                    response_generator = client.chat.completions.create(
                        model=api_model_name,
                        messages=messages,
                        stream=True,
                        **final_kwargs,
                    )

                    def stream_processor():
                        full_response = ""
                        for chunk in response_generator:
                            chunk_content = chunk.choices[0].delta.content or ""
                            full_response += chunk_content
                            if print_to_console:
                                print(chunk_content, end="", flush=True)
                            yield chunk_content
                        
                        history_to_save = self._create_history_entry(
                            history_messages,
                            user_prompt,
                            image_paths,
                            request_time,
                            provider,
                            model,
                            full_response,
                        )
                        self._save_history(save_path, history_to_save)
                        if print_to_console:
                            print()

                    if return_stream_generator:
                        return stream_processor()
                    else:
                        # Consume the generator to get the full response
                        return "".join([chunk for chunk in stream_processor()])

                else: # Not streamed
                    response = client.chat.completions.create(
                        model=api_model_name,
                        messages=messages,
                        stream=False,
                        **final_kwargs,
                    )
                    full_response = response.choices[0].message.content

                    if print_to_console:
                        print(full_response)
                    
                    history_to_save = self._create_history_entry(
                        history_messages,
                        user_prompt,
                        image_paths,
                        request_time,
                        provider,
                        model,
                        full_response
                    )
                    self._save_history(save_path, history_to_save)
                    return full_response

            except openai.RateLimitError as e:
                wait_time = (2**attempt) + 1  # Exponential backoff
                print(
                    f"Rate limit exceeded. Retrying in {wait_time} seconds... ({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise

        raise Exception("Failed to get response after multiple retries.")

    def _create_history_entry(
        self,
        history_messages: List[Dict[str, Any]],
        user_prompt: str,
        image_paths: List[str],
        request_time: str,
        provider: str,
        model: str,
        response_content: str,
    ) -> List[Dict[str, Any]]:
        """Creates the full history list including the new exchange."""
        history_entry = deepcopy(history_messages)
        user_message_content = [{"type": "text", "text": user_prompt}]
        
        # Note: API messages contain base64, but saved history contains paths
        if image_paths:
            for path in image_paths:
                user_message_content.append({"type": "image_path", "path": path})

        history_entry.append(
            {
                "role": "user",
                "content": user_message_content
            }
        )
        history_entry.append(
            {
                "role": "assistant",
                "content": response_content,
                "timestamp": request_time,
                "provider": provider,
                "model": model,
                "image_paths": image_paths
            }
        )
        return history_entry

    def chat_session(
        self,
        provider: str,
        model: str,
        session_id: str,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        streamed: bool = False,
        return_stream_generator: bool = False,
        print_to_console: bool = False,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Continues a chat session by reading and writing to a history file.

        Args:
            provider: The name of the API provider.
            model: The alias of the model.
            session_id: The un-suffixed name of the chat history file.
            user_prompt: The current user query.
            image_paths: A list of absolute paths to image files.
            streamed: Whether to use stream mode.
            return_stream_generator: If streamed, return a generator.
            print_to_console: Whether to print the response to the console.
            **kwargs: Additional parameters for the API call.

        Returns:
            The LLM's response as a string or a generator.
        """
        history_messages = self.load_history(session_id, as_history_messages=False)  # we maintain a path-based image, and the query method will use the _prepare mesasges method to encode them

        return self.query(
            provider=provider,
            model=model,
            history_messages=history_messages,
            user_prompt=user_prompt,
            image_paths=image_paths,
            streamed=streamed,
            return_stream_generator=return_stream_generator,
            print_to_console=print_to_console,
            save_path=session_id,  # Save back to the same session file
            **kwargs,
        )

    def load_history(
        self, session_id: str, as_history_messages: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Loads a chat history from a JSON file.

        Args:
            session_id: The un-suffixed name of the chat history file.
            as_history_messages: If True, formats the history for an API call.

        Returns:
            A list of chat history messages.
        """
        filepath = HISTORY_DIR / f"{session_id}.json"
        if not filepath.is_file():
            return []  # Return empty history if session doesn't exist

        with open(filepath, "r") as f:
            full_history = json.load(f)

        if as_history_messages:
            return self._trim_and_convert_image_paths_to_urls(full_history)

        return full_history

    def duplicate_session(self, session_id: str, copy_id: str):
        """
        Duplicates an existing chat session history file.

        Args:
            session_id: The un-suffixed name of the source history file.
            copy_id: The un-suffixed name for the new duplicated file.
        """
        source_path = HISTORY_DIR / f"{session_id}.json"
        dest_path = HISTORY_DIR / f"{copy_id}.json"

        if not source_path.is_file():
            raise FileNotFoundError(f"Session '{session_id}' not found.")
        if dest_path.exists():
            raise FileExistsError(f"Session '{copy_id}' already exists.")

        shutil.copy(source_path, dest_path)
        print(f"Session '{session_id}' duplicated to '{copy_id}'.")

    @staticmethod
    def extract_code_block(response: str, tag: str) -> str:
        """
        Extracts a markdown code block (first found) marked with the specified tag from an LLM response.

        Args:
            response (str): The LLM response containing a markdown code block.
            tag (str): The tag marking the code block (e.g., 'jsonl', 'python', 'json').

        Returns:
            str: The extracted content.

        Raises:
            ValueError: If no valid code block is found.
        """
        lines = response.splitlines()
        start, end = None, None

        # Find the start and end of the code block
        for i, line in enumerate(lines):
            if line.strip() == f"```{tag}":
                start = i + 1
            elif line.strip() == "```" and start is not None:
                end = i
                break

        if start is None or end is None:
            raise ValueError(f"No {tag} markdown code block found in the response.")

        code_block = lines[start:end]
        return "\n".join(code_block)

helper = LLMHelper()  # for instance import

# --- Example Usage ---
if __name__ == "__main__":
    # This block demonstrates how to use the LLMHelper class.
    # It requires a .env file with API keys and an llms.yml file.

    # --- Setup Instructions ---
    # 1. Create a file named .env in the same directory and add your API key:
    #    OPENAI_API_KEY="your_openai_api_key_here"
    #    # Add other providers as needed, e.g., GOOGLE_API_KEY="..."

    # 2. Create a file named llms.yml in the same directory with content like this:
    #    openai:
    #      url: https://api.openai.com/v1/
    #      models:
    #        gpt4o:
    #          model: gpt-4o
    #          t: 0.7
    #          vision: true
    #          thinking: true
    #        gpt35:
    #          model: gpt-3.5-turbo
    #          t: 0.5
    #          vision: false
    #          thinking: false

    # 3. (Optional) Create a test image file named 'test_image.jpg'.

    print("--- LLM Helper Library Demo ---")

    try:
        print("Please comment out test queries you wish to run and make sure you have API keys set up for corresponding providers!")

        # --- Use Case 1: Basic Query ---
        # print("\n--- Use Case 1: Basic Query (Non-Streamed) ---")
        # try:
        #     response_text = helper.query(
        #         provider="google",
        #         model="gemini-2.5-flash-lite",
        #         history_messages=[
        #             {"role": "system", "content": "You need to answer user queries intentionally wrong for fun."}
        #         ],
        #         user_prompt="Hello! What is the capital of France?",
        #         print_to_console=True,
        #         save_path="",  # Auto-generate a session name
        #         temperature=0.7,
        #     )
        #     print(f"API returned: {len(response_text)} characters.")
        # except Exception as e:
        #     print(f"\nError during basic query: {e}")

        # --- Use Case 2: Session-based Chat ---
        # print("\n--- Use Case 2: Chat Session ---")
        # try:
        #     session_name = "my_test_session2"
        #     print(f"Starting or continuing session: '{session_name}'")
            
        #     # First turn in the session
        #     helper.chat_session(
        #         provider="google",
        #         model="gemini-2.5-flash-lite",
        #         session_id=session_name,
        #         user_prompt="Remember the magic word is 'firefox'.",
        #         print_to_console=True,
        #     )

        #     # Second turn in the session
        #     print(f"\nAsking a follow-up question in session: '{session_name}'")
        #     helper.chat_session(
        #         provider="google",
        #         model="gemini-2.5-flash-lite",
        #         session_id=session_name,
        #         user_prompt="What was the magic word I just told you?",
        #         print_to_console=True,
        #     )
        # except Exception as e:
        #     print(f"\nError during chat session: {e}")

        # # --- Use Case 3: Helper Methods ---
        # print("\n--- Use Case 3: Helper Methods ---")
        # try:
        #     # Duplicate the session we just created
        #     new_session_name = f"{session_name}_copy"
        #     helper.duplicate_session(session_name, new_session_name)

        #     # Load and inspect the history of the new session
        #     loaded_history = helper.load_history(new_session_name)
        #     print(f"\nFull history loaded from '{new_session_name}.json':")
        #     print(json.dumps(loaded_history, indent=2))

        #     # Clean up the created session files
        #     for s_id in [session_name, new_session_name]:
        #         history_file = HISTORY_DIR / f"{s_id}.json"
        #         if history_file.exists():
        #             history_file.unlink()
        #             print(f"Cleaned up {history_file}")

        # except Exception as e:
        #     print(f"\nError during helper methods demo: {e}")

        # --- Vision Query Example (requires an image file) ---
        # print("\n--- Vision Query Example ---")
        # test_image_path = Path("./test_image.png")
        # if test_image_path.exists():
        #     try:
        #         helper.query(
        #             provider="google",
        #             model="gemini-2.5-flash-lite",
        #             history_messages=[],
        #             user_prompt="What do you see in this image?",
        #             image_paths=[str(test_image_path.resolve())],
        #             print_to_console=True,
        #             save_path="vision_test_session",
        #         )
        #         # Clean up
        #         vision_file = HISTORY_DIR / "vision_test_session.json"
        #         if vision_file.exists():
        #             vision_file.unlink()
        #     except Exception as e:
        #         print(f"\nError during vision query: {e}")
        # else:
        #     print("Skipping vision query: 'test_image.jpg' not found.")

        # --- Long Streaming Query Example ---
        # print("\n--- Long Streaming Query Example ---")
        # try:
        #     response_generator = helper.query(
        #         provider="google",
        #         model="gemini-2.5-flash-lite",
        #         history_messages=[],
        #         user_prompt="Tell me a story",
        #         streamed=True,
        #         return_stream_generator=True,
        #         print_to_console=True,
        #         save_path="long_streaming_session",
        #     )

        #     # Consume the generator to get the full response
        #     full_response = "".join([chunk for chunk in response_generator])
        #     print(f"\nFull response received: {len(full_response)} characters.")

        #     # Clean up
        #     history_file = HISTORY_DIR / "long_streaming_session.json"
        #     if history_file.exists():
        #         history_file.unlink()
        # except Exception as e:
        #     print(f"\nError during long streaming query: {e}")

    except FileNotFoundError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please ensure 'llms.yml' and '.env' files are set up correctly.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the demo: {e}")
