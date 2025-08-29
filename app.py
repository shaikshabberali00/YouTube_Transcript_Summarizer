# Importing Libraries
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import JSONFormatter
from urllib.parse import urlparse, parse_qs
from textwrap import dedent
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from gtts import gTTS
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
import os
os.environ["STREAMLIT_WATCHDOG"] = "false"

# Seed for reproducibility
DetectorFactory.seed = 0

# Streamlit Configuration
st.set_page_config(
    page_title="YouTube Summariser",
    page_icon='favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------- FUNCTIONS -------- #

# Spacy Extractive Summarizer
def spacy_summarize(text_content, percent):
    stop_words = list(STOP_WORDS)
    punctuation_items = punctuation + '\n'
    nlp = spacy.load('en_core_web_sm')

    nlp_object = nlp(text_content)
    word_frequencies = {}
    for word in nlp_object:
        if word.text.lower() not in stop_words and word.text.lower() not in punctuation_items:
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1

    if not word_frequencies:
        return ""

    max_frequency = max(word_frequencies.values())
    word_frequencies = {word: freq/max_frequency for word, freq in word_frequencies.items()}

    sentence_token = [sentence for sentence in nlp_object.sents]
    sentence_scores = {}
    for sent in sentence_token:
        for word in sent.text.split(" "):
            if word.lower() in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.lower()]

    select_length = int(len(sentence_token) * (int(percent) / 100))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    return ' '.join([sent.text for sent in summary])

# BART Abstractive Summarizer
def bart_summarize(text_content, max_length=150):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer.encode("summarize: " + text_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=max_length//2,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Split text
MAX_QUERY_LENGTH = 500
def split_text(text, chunk_size=MAX_QUERY_LENGTH):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Detect & Translate
def detect_and_translate(text, target_language):
    chunks = split_text(text)
    translated_chunks = []
    for chunk in chunks:
        try:
            lang_code = detect(chunk)
            translated = GoogleTranslator(source=lang_code, target=target_language).translate(chunk)
            translated_chunks.append(translated)
        except Exception:
            translated_chunks.append(chunk)  # fallback: original text
    return ' '.join(translated_chunks), text

# Extract YouTube ID
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        if parsed_url.path.startswith(('/embed/', '/v/')):
            return parsed_url.path.split('/')[2]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

# Fetch transcript safely
def generate_transcript(video_id, preferred_lang="en"):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[preferred_lang])
        script = '\n'.join([t['text'] for t in transcript])
        return script, len(script.split())
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            # fallback to auto-generated or any available transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            script = '\n'.join([t['text'] for t in transcript])
            return script, len(script.split())
        except:
            return None, 0

# Generate audio
def generate_audio(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("summary.mp3")
        with open("summary.mp3", "rb") as f:
            return f.read()
    except:
        return None

# --------- UI / APP --------- #

# Hide Streamlit Footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar logo (replace with your own)
file_ = open("app_logo.gif","rb")
data_url = base64.b64encode(file_.read()).decode("utf-8")
file_.close()
st.sidebar.markdown(f'<img src="data:image/gif;base64,{data_url}" width="300">', unsafe_allow_html=True)

# Video URL input
url = st.sidebar.text_input("Video URL", "https://www.youtube.com/watch?v=T-JVpKku5SI")
sumtype = st.sidebar.selectbox("Summarization Type", ["Extractive", "Abstractive (Subtitles)"])
language_options = {'English':'en', 'Telugu':'te','Hindi':'hi'}
target_language = st.sidebar.selectbox("Select Language", list(language_options.keys()))

# Display video
try:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("title").text.replace("&amp;", "&")
    st.info("### "+title)
    st.video(url)
except:
    st.warning("Could not fetch video info.")

# Main logic
if st.sidebar.button("Summarize"):
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL")
    else:
        transcript, _ = generate_transcript(video_id)
        if not transcript:
            st.warning("Transcript is disabled or unavailable")
        else:
            translated, original = detect_and_translate(transcript, language_options[target_language])
            if sumtype == "Extractive":
                summary = spacy_summarize(translated, 30)
            else:
                summary = bart_summarize(translated, 150)

            st.markdown(f"""
            <div style='background-color:#ADD8E6;padding:20px;border-radius:10px;'>
                <h3>📖 Summary / Subtitles</h3>
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)

            audio_bytes = generate_audio(summary, lang=language_options[target_language])
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
