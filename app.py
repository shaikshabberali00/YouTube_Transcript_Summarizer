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
import os
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator


# Set a seed value for reproducibility (optional)
DetectorFactory.seed = 0

# Streamlit Configuration
st.set_page_config(
    page_title="Youtube Summariser",
    page_icon='favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to Summarize using Spacy
def spacy_summarize(text_content, percent):
    stop_words = list(STOP_WORDS)
    punctuation_items = punctuation + '\n'
    nlp = spacy.load('en_core_web_sm')

    nlp_object = nlp(text_content)
    word_frequencies = {}
    for word in nlp_object:
        if word.text.lower() not in stop_words and word.text.lower() not in punctuation_items:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    if not word_frequencies:
        st.error("No valid words found for summarization.")
        return ""

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_token = [sentence for sentence in nlp_object.sents]
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.text.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    if not sentence_scores:
        st.error("No valid sentences found for summarization.")
        return ""

    select_length = int(len(sentence_token) * (int(percent) / 100))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary

# Function to Summarize using BART
def bart_summarize(text_content, max_length=150):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer.encode("summarize: " + text_content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=max_length//2, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Define the maximum allowed query length
MAX_QUERY_LENGTH = 500
def split_text(text, chunk_size=MAX_QUERY_LENGTH):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def detect_and_translate(text, target_language):
    if not isinstance(text, str) or not text.strip():
        st.error("Invalid input text.")
        return "", ""
    
    chunks = split_text(text)
    translated_chunks = []
    
    for chunk in chunks:
        try:
            # Detect the language of the chunk
            lang_code = detect(chunk)
            
            # Translate the chunk to the target language if it's not already in that language
            translated_text = GoogleTranslator(source=lang_code, target=target_language).translate(chunk)
            
            if translated_text and isinstance(translated_text, str) and translated_text.strip():
                translated_chunks.append(translated_text)
            else:
                st.error("Translation result is invalid or empty.")
                return "", text
            
        except LangDetectException as e:
            st.error(f"Error detecting language: {e}")
            return "", text
        except Exception as e:
            st.error(f"Error during translation: {e}")
            return "", text
    
    # Combine translated chunks into a single string
    full_translation = ' '.join(translated_chunks)
    return full_translation, text

# Hide Streamlit Footer and buttons
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.css-1l02zno {padding: 0 !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Adding logo for the App
file_ = open("app_logo.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.sidebar.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="" style="height:300px; width:400px;">',
    unsafe_allow_html=True,
)

# Set background color
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .summary-container {
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 20px;
        background-color: #ADD8E6;
        margin-bottom: 20px;
    }
    .summary-container h3 {
        color: #333;
    }
    .summary-container p {
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input Video Link
url = st.sidebar.text_input('Video URL', 'https://www.youtube.com/watch?v=T-JVpKku5SI')

# Display Video and Title
try:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")
    link = soup.find_all(name="title")[0]
    title = str(link).replace("<title>","").replace("</title>","").replace("&amp;","&")
    st.info("### " + title)
    st.video(url)
except Exception as e:
    st.error(f"Error fetching video details: {e}")

# Specify Summarization type
sumtype = st.sidebar.selectbox(
    'Specify Summarization Type',
    options=['Extractive', 'Abstractive (Subtitles)'],
    index=0,
    disabled=False
)

# Specify Translation Language
language_options = {
    'English': 'en',
    'Telugu': 'te',
    'Hindi': 'hi'
}
target_language = st.sidebar.selectbox(
    'Select Language',
    options=list(language_options.keys()),
    index=0,
    disabled=False
)

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
    elif parsed_url.hostname in ['youtu.be']:
        return parsed_url.path[1:]
    return None

def list_available_transcripts(video_id):
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        return transcripts
    except Exception as e:
        return None

def generate_transcript(video_id):
    global transcript_disabled_flag
    try:
        # Fetch available transcripts
        transcripts = list_available_transcripts(video_id)
        if not transcripts:
            if not transcript_disabled_flag:
                st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")
                transcript_disabled_flag = True
            return None, 0
        
        # Attempt to fetch English transcript
        try:
            transcript = transcripts.find_transcript(['en'])
        except:
            # If English transcript is not available, try other languages
            transcript = transcripts.find_transcript(['hi'])
        
        # Convert transcript to text
        formatter = JSONFormatter()
        transcript_json = transcript.fetch()
        
        # Check if transcript_json is valid
        if not transcript_json:
            if not transcript_disabled_flag:
                st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")
                transcript_disabled_flag = True
            return None, 0
        
        # Normalize the transcript JSON
        script = '\n'.join([item['text'] for item in transcript_json])
        
        if isinstance(script, str):
            return script, len(script.split())
        else:
            if not transcript_disabled_flag:
                st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")
                transcript_disabled_flag = True
            return None, 0
    
    except TranscriptsDisabled:
        if not transcript_disabled_flag:
            st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")
            transcript_disabled_flag = True
        return None, 0
    except NoTranscriptFound:
        if not transcript_disabled_flag:
            st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")
            transcript_disabled_flag = True
        return None, 0
    except Exception as e:
        st.error(f"An error occurred while generating the transcript: {e}")
        return None, 0

transcript_disabled_flag = False  # Initialize the flag

def generate_audio(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("summary.mp3")
        with open("summary.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

if sumtype == 'Extractive':
    # Specify the summary length
    length = st.sidebar.select_slider(
        'Specify length of Summary',
        options=['10%', '20%', '30%', '40%', '50%']
    )

    if st.sidebar.button('Summarize'):
        progress = st.progress(0)
        progress.progress(10)  # Initial progress

        video_id = extract_video_id(url)
        if video_id is None:
            st.error("Invalid YouTube URL. Please provide a valid URL.")
        else:
            progress.progress(20)  # Progress after extracting video ID

            transcript, _ = generate_transcript(video_id)
            if transcript:
                progress.progress(50)  # Progress after fetching transcript

                translated_transcript, original_transcript = detect_and_translate(transcript, language_options[target_language])
                if translated_transcript:
                    progress.progress(70)  # Progress after translation

                    summ = spacy_summarize(translated_transcript, int(length[:2]))
                    if summ.strip():
                        st.markdown(dedent(f"""
                        <div class="summary-container">
                            <h3>\U0001F4D6 Summary</h3>
                            <p>{summ}</p>
                        </div>
                        """), unsafe_allow_html=True)

                        progress.progress(90)  # Progress before generating audio
                        audio_bytes = generate_audio(summ, lang=language_options[target_language])
                        if audio_bytes:
                            st.audio(audio_bytes, format='audio/mp3')
                            progress.progress(100)  # Final progress
                        else:
                            st.error("Failed to generate audio for the summary.")
                    else:
                        st.error("Generated summary is empty.")
                else:
                    st.markdown(dedent(f"""
                    <div class="summary-container">
                        <h3>\U0001F4D6 Summary (Original Transcript)</h3>
                        <p>{original_transcript}</p>
                    </div>
                    """), unsafe_allow_html=True)
            else:
                if not transcript_disabled_flag:
                    st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")

elif sumtype == 'Abstractive (Subtitles)':
    if st.sidebar.button('Summarize'):
        progress = st.progress(0)
        progress.progress(10)  # Initial progress

        video_id = extract_video_id(url)
        if video_id is None:
            st.error("Invalid YouTube URL. Please provide a valid URL.")
        else:
            progress.progress(20)  # Progress after extracting video ID

            transcript, _ = generate_transcript(video_id)
            if transcript:
                progress.progress(50)  # Progress after fetching transcript

                translated_transcript, original_transcript = detect_and_translate(transcript, language_options[target_language])
                if translated_transcript:
                    progress.progress(70)  # Progress after translation

                    st.markdown(dedent(f"""
                    <div class="summary-container">
                        <h3>\U0001F3A5 Subtitles</h3>
                        <p>{translated_transcript}</p>
                    </div>
                    """), unsafe_allow_html=True)
                    
                    progress.progress(100)  # Final progress
                else:
                    st.markdown(dedent(f"""
                    <div class="summary-container">
                        <h3>\U0001F3A5 Subtitles (Original Transcript)</h3>
                        <p>{original_transcript}</p>
                    </div>
                    """), unsafe_allow_html=True)
            else:
                if not transcript_disabled_flag:
                    st.warning("TRANSCRIPT IS DISABLED FOR THIS VIDEO")
