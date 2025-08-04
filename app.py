import gradio as gr
from src.AgenticRAG.gpt import gpt_response
from src.SpeechToText.sr import transcribe_audio, clear_history
from src.SpeechToText.hamsa import transcribe_audio_hamsa
from datetime import datetime
from loguru import logger


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
            
            gpt_output = gr.Textbox(
                label="AI Therapeutic Response", 
                placeholder="AI response will appear here...",
                lines=5,
                rtl=True,
                interactive=False
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
    
    # Function to process transcription and get GPT response
    def process_audio_and_respond(audio, language, history):
        # Get transcription
        try:
            updated_history, current_text = transcribe_audio(audio, language, history)
            logger.info(f"Transcription successful: {current_text}")
        except Exception as e:
            updated_history, current_text = transcribe_audio_hamsa(audio, language, history)
            logger.error(f"Transcription failed. Apply Fallback with Hamsa API: {e}")
            if not current_text:
                current_text = "Transcription failed. Please try again."
        
        # Get GPT response if there's transcribed text
        gpt_result = ""
        if current_text and current_text.strip():
            response = gpt_response(current_text)
            gpt_result = f"Response: {response['response']} \n\nEmotion: {response['emotional_state']}"
            
            # Update history with both query and answer
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detected_lang = response.get('detected_language', 'Unknown')
            
            # Format the history entry
            history_entry = f"[{timestamp}] [{language}] [{detected_lang}]\n"
            history_entry += f"Query: {current_text}\n"
            history_entry += f"Answer: {response['response']}"
            history_entry += "-----------------------\n\n"
            
            # Add to history
            if updated_history:
                updated_history = history_entry + updated_history
            else:
                updated_history = history_entry
        
        return updated_history, current_text, gpt_result
    
    # Event handlers
    submit_btn.click(
        fn=process_audio_and_respond,
        inputs=[audio_input, language_selector, history_state],
        outputs=[history_state, current_output, gpt_output]
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[history_output]
    )
    
    clear_btn.click(
        fn=clear_history,
        outputs=[history_state, history_output]
    )
    
    # Auto-submit when audio is uploaded/recorded
    audio_input.change(
        fn=process_audio_and_respond,
        inputs=[audio_input, language_selector, history_state],
        outputs=[history_state, current_output, gpt_output]
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[history_output]
    )

if __name__ == "__main__":
    iface.launch(share=True)