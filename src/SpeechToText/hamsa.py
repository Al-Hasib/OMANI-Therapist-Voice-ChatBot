import requests
import base64
from datetime import datetime
import json
from dotenv import load_dotenv
import os
load_dotenv()


def transcribe_audio_hamsa(audio, language, history):
    """
    Transcribe audio using Hamsa API
    
    Args:
        audio: Audio file path or audio data
        language: Selected language from dropdown
        history: Previous transcription history
        api_key: Hamsa API key
    
    Returns:
        tuple: (updated_history, transcribed_text)
    """
    api_key = os.getenv("HAMS_API_KEY")
    if not api_key:
        raise ValueError("HAMS_API_KEY not set in environment variables")

    if audio is None:
        return history, ""
    
    # Language codes for Hamsa API
    language_codes = {
        "English": "en",
        "Arabic": "ar",
        "Arabic (Egypt)": "ar",
        "Arabic (UAE)": "ar",
        "Arabic (Lebanon)": "ar",
        "Arabic (Saudi Arabia)": "ar",
        "Arabic (Kuwait)": "ar",
        "Arabic (Qatar)": "ar",
        "Arabic (Jordan)": "ar",
        "Auto-detect": "auto"  # You may need to check if Hamsa supports auto-detection
    }
    
    try:
        # Convert audio file to base64
        if isinstance(audio, str):  # If audio is a file path
            with open(audio, 'rb') as audio_file:
                audio_bytes = audio_file.read()
        else:  # If audio is already bytes
            audio_bytes = audio
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Get selected language code
        selected_language = language_codes.get(language, "ar")
        
        # Prepare API request
        url = "https://api.tryhamsa.com/v1/realtime/stt"
        payload = {
            "audioList": [],  # Empty for single audio file
            "audioBase64": audio_base64,
            "language": selected_language,
            "isEosEnabled": False,
            "eosThreshold": 0.3
        }
        
        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }
        
        # Make API request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse response
        result = response.json()
        text = result.get("text", "")
        
        # Handle auto-detection result formatting
        if language == "Auto-detect" and text:
            # You might want to add language detection info if Hamsa provides it
            text = f"[Auto-detected] {text}"
        
        # Add timestamp and transcription to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp}] [{language}] {text}"
        
        # Update history
        if history:
            updated_history = history + "\n" + new_entry
        else:
            updated_history = new_entry
            
        return updated_history, text
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {e}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp}] [{language}] ERROR: {error_msg}"
        
        if history:
            updated_history = history + "\n" + new_entry
        else:
            updated_history = new_entry
            
        return updated_history, error_msg
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse API response: {e}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp}] [{language}] ERROR: {error_msg}"
        
        if history:
            updated_history = history + "\n" + new_entry
        else:
            updated_history = new_entry
            
        return updated_history, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp}] [{language}] ERROR: {error_msg}"
        
        if history:
            updated_history = history + "\n" + new_entry
        else:
            updated_history = new_entry
            
        return updated_history, error_msg

def clear_history():
    """Clear the transcription history"""
    return "", ""

# Example usage:
# api_key = "your-hamsa-api-key-here"
# history, text = transcribe_audio("path/to/audio.wav", "Arabic", "", api_key)
# print(f"Transcribed text: {text}")
# print(f"History: {history}")