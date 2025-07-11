import gradio as gr
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

def translate_text(text, target_language):
    """Simple translation function - you can integrate with Google Translate API"""
    try:
        from googletrans import Translator
        translator = Translator()
        
        if target_language == "Arabic":
            result = translator.translate(text, dest='ar')
            return result.text
        elif target_language == "English":
            result = translator.translate(text, dest='en')
            return result.text
        else:
            return text
    except:
        return "Translation service not available"

# Create Gradio Interface
with gr.Blocks(title="Multilingual Speech to Text") as iface:
    gr.Markdown("# 🎙️ Multilingual Speech to Text (Arabic & English)")
    gr.Markdown("Speak in Arabic or English, or let the system auto-detect the language!")
    
    with gr.Row():
        with gr.Column(scale=1):
            language_selector = gr.Dropdown(
                choices=[
                    "English", 
                    "Arabic", 
                    "Arabic (Egypt)", 
                    "Arabic (UAE)", 
                    "Arabic (Lebanon)",
                    "Arabic (Saudi Arabia)",
                    "Arabic (Kuwait)",
                    "Arabic (Jordan)",
                    "Auto-detect"
                ],
                value="Auto-detect",
                label="Select Language"
            )
            
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="🎤 Speak or Upload Audio"
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔄 Transcribe", variant="primary")
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
        
        with gr.Column(scale=1):
            current_output = gr.Textbox(
                label="Current Transcription", 
                placeholder="Your transcribed text will appear here...",
                lines=3,
                rtl=True  # Right-to-left for Arabic text
            )
            
            # Translation section
            with gr.Row():
                translate_btn = gr.Button("🌐 Translate to English", size="sm")
                translate_ar_btn = gr.Button("🌐 Translate to Arabic", size="sm")
            
            translated_output = gr.Textbox(
                label="Translation", 
                placeholder="Translations will appear here...",
                lines=2,
                rtl=True
            )
            
            history_output = gr.Textbox(
                label="Conversation History", 
                placeholder="All transcriptions will be saved here with timestamps...",
                lines=10,
                max_lines=20,
                interactive=False
            )
    
    # State to maintain history
    history_state = gr.State("")
    
    # Event handlers
    submit_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, language_selector, history_state],
        outputs=[history_state, current_output]
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[history_output]
    )
    
    clear_btn.click(
        fn=clear_history,
        outputs=[history_state, history_output]
    )
    
    # Translation buttons
    translate_btn.click(
        fn=lambda text: translate_text(text, "English"),
        inputs=[current_output],
        outputs=[translated_output]
    )
    
    translate_ar_btn.click(
        fn=lambda text: translate_text(text, "Arabic"),
        inputs=[current_output],
        outputs=[translated_output]
    )
    
    # Auto-submit when audio is uploaded/recorded
    audio_input.change(
        fn=transcribe_audio,
        inputs=[audio_input, language_selector, history_state],
        outputs=[history_state, current_output]
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[history_output]
    )

if __name__ == "__main__":
    iface.launch(share=True)