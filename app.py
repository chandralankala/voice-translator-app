import streamlit as st
import speech_recognition as sr
import openai
from langdetect import detect
from google.cloud import translate_v2 as translate
from google.cloud import language_v1
from google.cloud import texttospeech
import tempfile
import os

# Load API keys from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Google Clients
translate_client = translate.Client()
language_client = language_v1.LanguageServiceClient()
tts_client = texttospeech.TextToSpeechClient()

st.title("🎙️ Voice Translator with Emotion & Meaning Detection")

# Record audio
st.info("Click below to record your voice.")

audio_bytes = st.audio_input("Record your message", type="wav")

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name

    st.audio(audio_path)

    # Recognize speech
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Detected Speech: {text}")
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
            st.stop()

    # Language detection
    source_lang = detect(text)
    st.write(f"Detected Language: `{source_lang}`")

    # Choose target language
    target_lang = st.selectbox("Translate to:", ["en", "es", "fr", "de", "hi", "zh"])

    # Translate
    translated = translate_client.translate(text, target_language=target_lang)
    st.success(f"Translated Text: {translated['translatedText']}")

    # Sentiment Analysis
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = language_client.analyze_sentiment(request={'document': document}).document_sentiment

    mood = "😐 Neutral"
    if sentiment.score > 0.3:
        mood = "😊 Positive"
    elif sentiment.score < -0.3:
        mood = "😠 Negative"

    st.write(f"Sentiment: {mood} (score: {sentiment.score:.2f})")

    # Meaning Analysis via OpenAI
    gpt_prompt = f"Analyze this phrase and explain what the speaker really means (context, emotion, intent):\n\"{text}\""
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": gpt_prompt}]
    )
    underlying_meaning = gpt_response.choices[0].message.content.strip()
    st.write("🧠 Underlying Meaning:")
    st.info(underlying_meaning)

    # Text-to-speech
    synthesis_input = texttospeech.SynthesisInput(text=translated["translatedText"])
    voice = texttospeech.VoiceSelectionParams(
        language_code=target_lang if target_lang != "zh" else "cmn-CN",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(tts_path, "wb") as out:
        out.write(response.audio_content)

    st.audio(tts_path)
