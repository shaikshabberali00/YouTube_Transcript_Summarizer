# Importing Libraries
import streamlit as st
import base64
from bs4 import BeautifulSoup
import requests
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

# Set a seed value for reproducibility
DetectorFactory.seed = 0

# Streamlit Configuration
st.set_page_config(
    page_title="Youtube Summariser",
    page_icon='favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------- FUNCTIONS -------- #

# Function to Summarize using Spacy
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
        st.error("No valid words found for summarization.")
        return ""

    max_frequency = max(word_frequencies.values())
    word_frequencies = {word: freq/max_frequency for word, freq in word_frequencies.items()}

    sentence_token = [sentence for sentence in nlp_object.sents]
    sentence_scores = {}
    for sent in sentence_token:
        for word in sent.text.split(" "):
            if word.lower() in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.lower()]

    if not sentence_scores:
        st.error("No valid sentences found for summarization.")
        return ""

    select_length = int(len(sentence_token) * (int(percent) / 100))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    return ' '.join(final_summary)


# Function to Summarize using BART
def bart_summarize(text_content, max_length=150):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer.encode("summarize: " + text_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=max_length//2,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Split text into chunks
MAX_QUERY_LENGTH = 500
def split_text(text, chunk_size=MAX_QUERY_LENGTH):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Detect & Translate
def detect_and_translate(text, target_language):
    if not isinstance(text, str) or not text.strip():
        st.error("Invalid input text.")
        return "", ""
    
    chunks = split_text(text)
    translated_chunks = []
    
    for chunk in chunks:
        try:
            lang_code = detect(chunk)
            translated_text = GoogleTranslator(source=lang_code, target=target_language).translate(chunk)
            if translated_text and isinstance(translated_text, str) and translated_text.strip():
                translated_chunks.append(translated_text)
            else:
                return "", text
        except LangDetectException as e:
            st.error(f"Error detecting language: {e}")
            return "", text
        except Exception as e:
            st.error(f"Error during translation: {e}")
            return "", text
    
    return ' '.join(translated_chunks), text


# Extract video ID
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None


# ✅ Improved Transcript Fetcher
def generate_transcript(video_id, preferred_lang="en"):
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = None
        # 1. Try exact match
        try:
            transcript = transcripts.find_transcript([preferred_lang])
        except:
            pass

        # 2. Try variations like en-GB, en-US
        if not transcript:
            try:
                transcript = transcripts.find_transcript(
                    [lang for lang in transcripts._manually_created_transcripts.keys() if lang.startswith(preferred_lang)]
                )
            except:
                pass

        # 3. Try auto-generated English transcripts
        if not transcript:
            try:
                transcript = transcripts.find_transcript(
                    [lang for lang in transcripts._generated_transcripts.keys() if lang.startswith(preferred_lang)]
                )
            except:
                pass

        # 4. Fallback: any available transcript
        if not transcript:
            transcript = list(transcripts)[0]

        transcript_json = transcript.fetch()
        script = '\n'.join([item['text'] for item in transcript_json])
        return script, len(script.split())

    except (TranscriptsDisabled, NoTranscriptFound):
        return None, 0
    except Exception as e:
        st.error(f"Transcript error: {e}")
        return None, 0


# Generate audio
def generate_audio(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("summary.mp3")
        with open("summary.mp3", "rb") as audio_file:
            return audio_file.read()
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None


# --------- UI / APP --------- #

# Hide Streamlit Footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.css-1l02zno {padding: 0 !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Logo
file_ = open("app_logo.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.sidebar.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="" style="height:300px; width:400px;">',
    unsafe_allow_html=True,
)

# Input Video Link
url = st.sidebar.text_input('Video URL', 'https://www.youtube.com/watch?v=T-JVpKku5SI')

# Display Video and Title
try:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("title").text.replace("&amp;", "&")
    st.info("### " + title)
    st.video(url)
except Exception as e:
    st.error(f"Error fetching video details: {e}")

# Summarization Type
sumtype = st.sidebar.selectbox(
    'Specify Summarization Type',
    options=['Extractive', 'Abstractive (Subtitles)'],
    index=0
)

# Translation Language
language_options = {
    'English': 'en',
    'Telugu': 'te',
    'Hindi': 'hi'
}
target_language = st.sidebar.selectbox(
    'Select Language',
    options=list(language_options.keys()),
    index=0
)

transcript_disabled_flag = False

# -------- Main Logic -------- #
if sumtype == 'Extractive':
    length = st.sidebar.select_slider(
        'Specify length of Summary',
        options=['10%', '20%', '30%', '40%', '50%']
    )

    if st.sidebar.button('Summarize'):
        progress = st.progress(0)
        progress.progress(10)

        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            progress.progress(20)
            transcript, _ = generate_transcript(video_id)

            if transcript:
                progress.progress(50)
                translated_transcript, original_transcript = detect_and_translate(transcript, language_options[target_language])

                if translated_transcript:
                    progress.progress(70)
                    summ = spacy_summarize(translated_transcript, int(length[:2]))
                    if summ.strip():
                        st.markdown(f"""
                        <div class="summary-container">
                            <h3>📖 Summary</h3>
                            <p>{summ}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        progress.progress(90)
                        audio_bytes = generate_audio(summ, lang=language_options[target_language])
                        if audio_bytes:
                            st.audio(audio_bytes, format='audio/mp3')
                        progress.progress(100)
                else:
                    st.markdown(f"""
                    <div class="summary-container">
                        <h3>📖 Original Transcript</h3>
                        <p>{original_transcript}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")

elif sumtype == 'Abstractive (Subtitles)':
    if st.sidebar.button('Summarize'):
        progress = st.progress(0)
        progress.progress(10)

        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            progress.progress(20)
            transcript, _ = generate_transcript(video_id)

            if transcript:
                progress.progress(50)
                translated_transcript, original_transcript = detect_and_translate(transcript, language_options[target_language])
                if translated_transcript:
                    st.markdown(f"""
                    <div class="summary-container">
                        <h3>🎥 Subtitles</h3>
                        <p>{translated_transcript}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="summary-container">
                        <h3>🎥 Original Transcript</h3>
                        <p>{original_transcript}</p>
                    </div>
                    """, unsafe_allow_html=True)
                progress.progress(100)
            else:
                st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")
