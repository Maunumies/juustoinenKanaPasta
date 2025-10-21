"""
helpers.py - Helper functions for structured AI generation

This file contains utility functions that connect to OpenAI's API
and handle structured output generation using Pydantic models.
"""

import os
from openai import OpenAI
from pydantic import BaseModel
from typing import Type, TypeVar
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TypeVar allows us to maintain type hints for any Pydantic model
T = TypeVar('T', bound=BaseModel)

def structured_generator(model: str, prompt: str, response_model: Type[T]) -> T:
    """
    Generates structured output from OpenAI's API using Pydantic models.
    
    Args:
        model (str): The OpenAI model to use (e.g., "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo")
        prompt (str): The prompt/instruction for the AI
        response_model (Type[T]): A Pydantic BaseModel class that defines the expected output structure
    
    Returns:
        T: An instance of the response_model filled with AI-generated data
    
    Raises:
        ValueError: If API key is not set
        Exception: If API call fails
    
    Example:
        class MyModel(BaseModel):
            name: str
            age: int
        
        result = structured_generator("gpt-4", "Generate a person", MyModel)
        print(result.name)  # AI-generated name
    """
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found! Please set your OPENAI_API_KEY in the .env file.\n"
            "You can get an API key from: https://platform.openai.com/api-keys\n\n"
            "Create a .env file in your project directory with:\n"
            "OPENAI_API_KEY=your-key-here"
        )
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # Call OpenAI API with structured output parsing
        # This uses the beta.chat.completions.parse endpoint which enforces
        # that the response matches your Pydantic model structure
        
        # Build base parameters
        params = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that provides structured, accurate responses."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": response_model
        }
        
        # Only add temperature for models that support it (GPT-5 doesn't)
        if not model.startswith("gpt-5"):
            params["temperature"] = 0.7  # Controls randomness (0.0 = deterministic, 1.0 = creative)
        
        completion = client.beta.chat.completions.parse(**params)
        
        # Extract and return the parsed response
        # The API automatically validates it matches your Pydantic model
        return completion.choices[0].message.parsed
        
    except Exception as e:
        # Provide helpful error messages
        print(f"❌ Error calling OpenAI API: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that your API key is valid in the .env file")
        print("2. Verify you have API credits available")
        print("3. Ensure the model name is correct")
        print(f"4. Model requested: {model}")
        raise


def structured_generator_with_system_prompt(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T]
) -> T:
    """
    Same as structured_generator but allows custom system prompt.
    
    Use this when you want more control over the AI's behavior/personality.
    
    Args:
        model (str): The OpenAI model to use
        system_prompt (str): Instructions about how the AI should behave
        user_prompt (str): The actual user request/prompt
        response_model (Type[T]): Pydantic model for output structure
    
    Returns:
        T: Structured response matching the model
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file!")
    
    client = OpenAI(api_key=api_key)
    
    try:
        # Build base parameters
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": response_model
        }
        
        # Only add temperature for models that support it (GPT-5 doesn't)
        if not model.startswith("gpt-5"):
            params["temperature"] = 0.7
        
        completion = client.beta.chat.completions.parse(**params)
        
        return completion.choices[0].message.parsed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


# Optional: Function to test if API key is working
def test_api_connection() -> bool:
    """
    Tests if OpenAI API connection is working.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No API key found in .env file")
            return False
        
        client = OpenAI(api_key=api_key)
        
        # Make a minimal API call to test connection
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ API connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the API connection when this file is run directly
    print("Testing OpenAI API connection...")
    test_api_connection()