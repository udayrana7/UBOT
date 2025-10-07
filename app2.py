from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pytube import YouTube
import librosa
import soundfile as sf
import torch
import os

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Initialize the summarization model
summarizer = pipeline("summarization")

# Load wav2vec2 model & tokenizer once (English)
asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_with_wav2vec2(video_url):
    """Download audio and transcribe with wav2vec2 if captions are not available."""
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_path = "temp_audio.wav"
    audio_stream.download(filename=audio_path)

    speech, rate = librosa.load(audio_path, sr=16000)
    inputs = asr_tokenizer(speech, return_tensors="pt", padding="longest")

    with torch.no_grad():
        logits = asr_model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = asr_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return " ".join(transcription)

# Import LangChain and other dependencies
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys from environment variables
hf_token = os.getenv('HuggingFaceHub_API_Token')
google_key = os.getenv('GOOGLE_API_KEY')
cohere_key = os.getenv('cohere_api_key')

os.environ['HuggingFaceHub_API_Token'] = hf_token if hf_token else ''
os.environ['GOOGLE_API_KEY'] = google_key if google_key else ''
os.environ['cohere_api_key'] = cohere_key if cohere_key else ''

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Function to split text and create vector store
def setup_rag_system(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        separators=['\n', '\n\n', ' ', '']
    )
    chunks = text_splitter.split_text(text=text)

    # Indexing the data using FAISS
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Creating retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                not contained in the context, say "answer not available in context" \n\n
                Context: \n {context}?\n
                Question: \n {question} \n
                Answer:"""

    prompt = PromptTemplate.from_template(template=prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(question):
        cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key=os.getenv('cohere_api_key'))
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | cohere_llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
    
    return generate_answer

# Placeholder for the generate_answer function
generate_answer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    global generate_answer
    data = request.get_json()
    url = data.get('url', '')
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id)
        texts = [item['text'] for item in srt]
        combined_text = ' '.join(texts)
    except Exception:
        # Fallback: transcribe audio using wav2vec2
        combined_text = transcribe_with_wav2vec2(url)

    summary = summarizer(combined_text, max_length=500, min_length=100, do_sample=False, truncation=True)[0]['summary_text']

    # Set up the RAG system with the combined transcript text
    generate_answer = setup_rag_system(combined_text)

    generate_answer_str = "function(question) { return fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: question }) }).then(response => response.json()).then(data => data.answer); }"

    return jsonify({'transcript': combined_text, 'summary': summary, 'generate_answer': generate_answer_str})

@app.route('/ask', methods=['POST'])
def ask():
    global generate_answer
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if not generate_answer:
        return jsonify({'error': 'RAG system not initialized'}), 500

    try:
        answer = generate_answer(question)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'answer': answer})

def extract_video_id(url):
    start = url.find("youtu.be/")
    if start == -1:
        start = url.find("youtube.com/watch?v=")
        if start == -1:
            return None
        start += len("youtube.com/watch?v=")
    else:
        start += len("youtu.be/")
        
    end = url.find("&", start)
    if end == -1:
        end = len(url)
        
    return url[start:end]

if __name__ == '__main__':
    app.run(debug=True)
