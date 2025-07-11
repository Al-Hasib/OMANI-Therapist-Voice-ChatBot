import speech_recognition as sr
from datetime import datetime

def transcribe_audio(audio, language, history):
    recognizer = sr.Recognizer()
    
    if audio is None:
        return history, ""
    
    # Language codes for Google Speech Recognition
    language_codes = {
        "English": "en-US",
        "Arabic": "ar-SA",  # Saudi Arabic
        "Arabic (Egypt)": "ar-EG",
        "Arabic (UAE)": "ar-AE",
        "Arabic (Lebanon)": "ar-LB",
        "Arabic (Saudi Arabia)": "ar-SA",
        "Arabic (Kuwait)": "ar-KW",
        "Arabic (Qatar)": "ar-QA",
        "Arabic (Jordan)": "ar-JO",
        "Auto-detect": None  # Let Google auto-detect
    }
    
    try:
        with sr.AudioFile(audio) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
        
        # Get selected language code
        selected_language = language_codes.get(language, "en-US")
        
        # Transcribe based on language selection
        if selected_language:
            text = recognizer.recognize_google(audio_data, language=selected_language)
        else:
            # Auto-detect: try Arabic first, then English
            try:
                text = recognizer.recognize_google(audio_data, language="ar-SA")
                detected_lang = "Arabic"
            except:
                text = recognizer.recognize_google(audio_data, language="en-US")
                detected_lang = "English"
            
            text = f"[{detected_lang}] {text}"
        
        # Add timestamp and transcription to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp}] [{language}] {text}"
        
        # Update history
        if history:
            updated_history = history + "\n" + new_entry
        else:
            updated_history = new_entry
            
        return updated_history, text
        
    except sr.UnknownValueError:
        error_msg = "Could not understand audio"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp}] [{language}] ERROR: {error_msg}"
        
        if history:
            updated_history = history + "\n" + new_entry
        else:
            updated_history = new_entry
            
        return updated_history, error_msg
        
    except sr.RequestError as e:
        error_msg = f"Could not request results; {e}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp}] [{language}] ERROR: {error_msg}"
        
        if history:
            updated_history = history + "\n" + new_entry
        else:
            updated_history = new_entry
            
        return updated_history, error_msg

def clear_history():
    return "", ""