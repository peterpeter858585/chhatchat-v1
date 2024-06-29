import gradio as gr
import edge_tts
import asyncio
import tempfile
import numpy as np
import soxr
from pydub import AudioSegment
import torch
import sentencepiece as spm
import onnxruntime as ort
from huggingface_hub import hf_hub_download, InferenceClient
import requests
from bs4 import BeautifulSoup
import urllib
import random

# List of user agents to choose from for requests
_useragent_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.62',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0'
]

def get_useragent():
    """Returns a random user agent from the list."""
    return random.choice(_useragent_list)

def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    # Get the remaining visible text
    visible_text = soup.get_text(strip=True)
    return visible_text

def search(term, num_results=1, lang="en", advanced=True, sleep_interval=0, timeout=5, safe="active", ssl_verify=None):
    """Performs a Google search and returns the results."""
    escaped_term = urllib.parse.quote_plus(term)
    start = 0
    all_results = []

    # Fetch results in batches
    while start < num_results:
        resp = requests.get(
            url="https://www.google.com/search",
            headers={"User-Agent": get_useragent()}, # Set random user agent
            params={
                "q": term,
                "num": num_results - start, # Number of results to fetch in this batch
                "hl": lang,
                "start": start,
                "safe": safe,
            },
            timeout=timeout,
            verify=ssl_verify,
        )
        resp.raise_for_status() # Raise an exception if request fails

        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", attrs={"class": "g"})

        # If no results, continue to the next batch
        if not result_block:
            start += 1
            continue

        # Extract link and text from each result
        for result in result_block:
            link = result.find("a", href=True)
            if link:
                link = link["href"]
                try:
                    # Fetch webpage content
                    webpage = requests.get(link, headers={"User-Agent": get_useragent()})
                    webpage.raise_for_status()
                    # Extract visible text from webpage
                    visible_text = extract_text_from_webpage(webpage.text)
                    all_results.append({"link": link, "text": visible_text})
                except requests.exceptions.RequestException as e:
                    # Handle errors fetching or processing webpage
                    print(f"Error fetching or processing {link}: {e}")
                    all_results.append({"link": link, "text": None})
            else:
                all_results.append({"link": None, "text": None})

        start += len(result_block) # Update starting index for next batch

    return all_results

# Speech Recognition Model Configuration
model_name = "neongeckocom/stt_en_citrinet_512_gamma_0_25"
sample_rate = 16000

# Download preprocessor, encoder and tokenizer
preprocessor = torch.jit.load(hf_hub_download(model_name, "preprocessor.ts", subfolder="onnx"))
encoder = ort.InferenceSession(hf_hub_download(model_name, "model.onnx", subfolder="onnx"))
tokenizer = spm.SentencePieceProcessor(hf_hub_download(model_name, "tokenizer.spm", subfolder="onnx"))

# Mistral Model Configuration
client1 = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")
system_instructions1 = "<s>[SYSTEM] Answer as Real Jarvis JARVIS, Made by 'Tony Stark', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. The request asks you to provide friendly responses as if You are the character Jarvis, made by 'Tony Stark.' The expectation is that I will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"

def resample(audio_fp32, sr):
    return soxr.resample(audio_fp32, sr, sample_rate)

def to_float32(audio_buffer):
    return np.divide(audio_buffer, np.iinfo(audio_buffer.dtype).max, dtype=np.float32)

def transcribe(audio_path):
    audio_file = AudioSegment.from_file(audio_path)
    sr = audio_file.frame_rate
    audio_buffer = np.array(audio_file.get_array_of_samples())

    audio_fp32 = to_float32(audio_buffer)
    audio_16k = resample(audio_fp32, sr)

    input_signal = torch.tensor(audio_16k).unsqueeze(0)
    length = torch.tensor(len(audio_16k)).unsqueeze(0)
    processed_signal, _ = preprocessor.forward(input_signal=input_signal, length=length)
    
    logits = encoder.run(None, {'audio_signal': processed_signal.numpy(), 'length': length.numpy()})[0][0]

    blank_id = tokenizer.vocab_size()
    decoded_prediction = [p for p in logits.argmax(axis=1).tolist() if p != blank_id]
    text = tokenizer.decode_ids(decoded_prediction)

    return text

def model(text, web_search):
    if web_search is True:
        """Performs a web search, feeds the results to a language model, and returns the answer."""
        web_results = search(text)
        web2 = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])
        formatted_prompt = system_instructions1 + text + "[WEB]" + str(web2) + "[ANSWER]"
        stream = client1.text_generation(formatted_prompt, max_new_tokens=512, stream=True, details=True, return_full_text=False)
        return "".join([response.token.text for response in stream if response.token.text != "</s>"])
    else:
        formatted_prompt = system_instructions1 + text + "[JARVIS]"
        stream = client1.text_generation(formatted_prompt, max_new_tokens=512, stream=True, details=True, return_full_text=False)
        return "".join([response.token.text for response in stream if response.token.text != "</s>"])

async def respond(audio, web_search):
    user = transcribe(audio)
    reply = model(user, web_search)
    communicate = edge_tts.Communicate(reply)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    return tmp_path