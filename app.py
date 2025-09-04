# import gradio as gr
# from src.agenticRAG.gpt import gpt_response
# from src.SpeechToText.sr import transcribe_audio, clear_history
# from src.SpeechToText.hamsa import transcribe_audio_hamsa
# from datetime import datetime
# from loguru import logger
# from src.TextToSpeech.gtts_tts import text_to_speech_with_gtts

# # Create Gradio Interface
# with gr.Blocks(title="Multilingual Speech to Text") as iface:
#     gr.Markdown("# 🎙️ Multilingual Speech to Text (Arabic & English)")
#     gr.Markdown("Speak in Arabic or English, or let the system auto-detect the language!")
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             language_selector = gr.Dropdown(
#                 choices=[
#                     "English", 
#                     "Arabic", 
#                     "Arabic (Egypt)", 
#                     "Arabic (UAE)", 
#                     "Arabic (Lebanon)",
#                     "Arabic (Saudi Arabia)",
#                     "Arabic (Kuwait)",
#                     "Arabic (Jordan)",
#                     "Auto-detect"
#                 ],
#                 value="Auto-detect",
#                 label="Select Language"
#             )
            
#             audio_input = gr.Audio(
#                 sources=["microphone", "upload"], 
#                 type="filepath", 
#                 label="🎤 Speak or Upload Audio"
#             )
            
#             with gr.Row():
#                 submit_btn = gr.Button("🔄 Transcribe", variant="primary")
#                 clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
        
#         with gr.Column(scale=1):
#             current_output = gr.Textbox(
#                 label="Current Transcription", 
#                 placeholder="Your transcribed text will appear here...",
#                 lines=3,
#                 rtl=True  # Right-to-left for Arabic text
#             )
            
#             gpt_output = gr.Textbox(
#                 label="AI Therapeutic Response", 
#                 placeholder="AI response will appear here...",
#                 lines=5,
#                 rtl=True,
#                 interactive=False
#             )
            
#             history_output = gr.Textbox(
#                 label="Conversation History", 
#                 placeholder="All transcriptions will be saved here with timestamps...",
#                 lines=10,
#                 max_lines=20,
#                 interactive=False
#             )
    
#     # State to maintain history
#     history_state = gr.State("")
    
#     # Function to process transcription and get GPT response
#     def process_audio_and_respond(audio, language, history):
#         # Get transcription
#         try:
#             updated_history, current_text = transcribe_audio(audio, language, history)
#             logger.info(f"Transcription successful: {current_text}")
#         except Exception as e:
#             updated_history, current_text = transcribe_audio_hamsa(audio, language, history)
#             logger.error(f"Transcription failed. Apply Fallback with Hamsa API: {e}")
#             if not current_text:
#                 current_text = "Transcription failed. Please try again."
        
#         # Get GPT response if there's transcribed text
#         gpt_result = ""
#         if current_text and current_text.strip():
#             response = gpt_response(current_text)
#             gpt_result = f"Response: {response['response']} \n\nEmotion: {response['emotional_state']}"
            
#             # Update history with both query and answer
            
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             detected_lang = response.get('detected_language', 'Unknown')
            
#             # Format the history entry
#             history_entry = f"[{timestamp}] [{language}] [{detected_lang}]\n"
#             history_entry += f"Query: {current_text}\n"
#             history_entry += f"Answer: {response['response']}"
#             history_entry += "-----------------------\n\n"
            
#             # Add to history
#             if updated_history:
#                 updated_history = history_entry + updated_history
#             else:
#                 updated_history = history_entry
        
#         return updated_history, current_text, gpt_result
    
#     # Event handlers
#     submit_btn.click(
#         fn=process_audio_and_respond,
#         inputs=[audio_input, language_selector, history_state],
#         outputs=[history_state, current_output, gpt_output]
#     ).then(
#         fn=lambda h: h,
#         inputs=[history_state],
#         outputs=[history_output]
#     )
    
#     clear_btn.click(
#         fn=clear_history,
#         outputs=[history_state, history_output]
#     )
    
#     # Auto-submit when audio is uploaded/recorded
#     audio_input.change(
#         fn=process_audio_and_respond,
#         inputs=[audio_input, language_selector, history_state],
#         outputs=[history_state, current_output, gpt_output]
#     ).then(
#         fn=lambda h: h,
#         inputs=[history_state],
#         outputs=[history_output]
#     )

# if __name__ == "__main__":
#     # iface.launch(share=True)
#     demo = iface.launch(
#         share=True,
#         server_name="0.0.0.0",
#         server_port=7860,
#         show_error=True,
#         debug=True
#     )
    
#     if hasattr(demo, 'share_url') and demo.share_url:
#         logger.info(f"🌐 Share link: {demo.share_url}")
#     else:
#         logger.warning("❌ Share link not generated")


import gradio as gr
import os
import tempfile
from src.agenticRAG.gpt import gpt_response
from src.SpeechToText.sr import transcribe_audio, clear_history
from src.SpeechToText.hamsa import transcribe_audio_hamsa
from datetime import datetime
from loguru import logger
from src.TextToSpeech.gtts_tts import text_to_speech_with_gtts

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
            
            # Add TTS audio output
            tts_audio = gr.Audio(
                label="🔊 AI Response Audio",
                type="filepath",
                interactive=False,
                autoplay=True  # Automatically play the audio when generated
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
    
    # Function to generate TTS audio
    def generate_tts_audio(text):
        """Generate TTS audio and return the file path"""
        if not text or not text.strip():
            return None
        
        try:
            # Create a temporary file for the audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Generate TTS audio
            text_to_speech_with_gtts(text, temp_audio_path)
            
            return temp_audio_path
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
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
        tts_audio_path = None
        
        if current_text and current_text.strip():
            response = gpt_response(current_text)
            gpt_result = f"Response: {response['response']} \n\nEmotion: {response['emotional_state']}"
            
            # Generate TTS for the AI response
            ai_response_text = response['response']
            tts_audio_path = generate_tts_audio(ai_response_text)
            
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
        
        return updated_history, current_text, gpt_result, tts_audio_path
    
    # Function to clear everything including TTS audio
    def clear_all():
        return "", "", "", None
    
    # Event handlers
    submit_btn.click(
        fn=process_audio_and_respond,
        inputs=[audio_input, language_selector, history_state],
        outputs=[history_state, current_output, gpt_output, tts_audio]
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[history_output]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[history_state, history_output, current_output, gpt_output, tts_audio]
    )
    
    # Auto-submit when audio is uploaded/recorded
    audio_input.change(
        fn=process_audio_and_respond,
        inputs=[audio_input, language_selector, history_state],
        outputs=[history_state, current_output, gpt_output, tts_audio]
    ).then(
        fn=lambda h: h,
        inputs=[history_state],
        outputs=[history_output]
    )

if __name__ == "__main__":
    # iface.launch(share=True)
    demo = iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True
    )
    
    if hasattr(demo, 'share_url') and demo.share_url:
        logger.info(f"🌐 Share link: {demo.share_url}")
    else:
        logger.warning("❌ Share link not generated")