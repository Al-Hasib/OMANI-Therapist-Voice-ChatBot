from langdetect import detect, detect_langs

# Sample texts
texts = {
    "Arabic": "مرحبا كيف حالك؟ I love you",        # Arabic
    "English": "Hello, how are you?",   # English
    "Bangla": "তুমি কেমন আছো?",         # Bangla
}
import time
for lang_name, sentence in texts.items():
    start = time.time()
    detected_lang = detect(sentence)
    confidence = detect_langs(sentence)[0].prob
    end = time.time()
    print(f"{lang_name}:")
    print(f"  Text: {sentence}")
    print(f"  Detected Language Code: {detected_lang}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Time Taken: {end - start:.4f} seconds\n")
    print("=" * 50)


import langid
import time

texts = {
    "Arabic": "مرحبا كيف حالك؟ I love you",
    "English": "Hello, how are you?",
    "Bangla": "তুমি কেমন আছো?",
}

for lang_name, sentence in texts.items():
    start = time.time()
    lang, confidence = langid.classify(sentence)
    end = time.time()
    print(f"{lang_name}:")
    print(f"  Detected: {lang}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Time Taken: {end - start:.6f} seconds\n")
    print("=" * 50)

