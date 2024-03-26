import streamlit as st
import os
import PyPDF2
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from pytube import YouTube
from moviepy.editor import *
import openai
from langchain.text_splitter import  CharacterTextSplitter
import tiktoken
import hnswlib
import numpy as np

openai_params1 = {"model":"gpt-3.5-turbo-1106",
                 "temperature":0.5,
                 "frequency_penalty":0.0,
                 "presence_penalty":0.0,
                 "max_tokens":1500,
                 "top_p":1}

openai_params = {"model":"gpt-4-1106-preview",
                 "temperature":0.5,
                 "frequency_penalty":0.0,
                 "presence_penalty":0.0,
                 "max_tokens":1500,
                 "top_p":1}

def extract_text_from_pdf(pdf_file_path):
    # Open the PDF file
    with open(pdf_file_path, 'rb') as file:
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        extracted_text = ""

        for page in pdf_reader.pages:

            extracted_text += page.extract_text()

        return extracted_text


def scrape_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract all text from the webpage
            text = soup.get_text(separator='\n', strip=True)
            return text
        else:
            return f"Failed to retrieve content from {url}. Status code: {response.status_code}"
    except requests.RequestException as e:
        return f"Error during requests to {url} : {str(e)}"


# file_description
def file_description_prompt(text):
  prompt = f"""Your task is to craft a summary of the provided text, aiming for a length of 140-150 words. This compact summary should distill the text to its core elements and serve as a basis for formulating efficient search queries. Your summary should concisely include:

Main themes and subjects, encapsulated in a brief overview.
Key details, examples, or notable elements, described succinctly.
Relevant additional context or related topics, briefly touched upon.
Ensure that the summary is comprehensive yet concise, ideally within the 140-150 word range, to effectively capture the text's essence for subsequent search query development.

Text for Summarization:
{text}

140-150 Word Concise Summary:
"""
  return prompt

def count_tokens(input):

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(input))

def truncate_to_token_limit(input,limit):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(input)[:limit])

def generate_summary(text):
  if count_tokens(text) > 16000:
      text = truncate_to_token_limit(text,14000)
  else:
      text = text 
  prompt = file_description_prompt(text)
  message = [{"role":"user","content":prompt}]
  response = openai.ChatCompletion.create(messages=message,
                                        **openai_params1)

  return response.choices[0].message.content

def download_audio(youtube_url):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(filename="audio_temp")

    # Convert to MP3 using moviepy
    audio_clip = AudioFileClip(audio_file)
    mp3_filename = "audio.mp3"
    audio_clip.write_audiofile(mp3_filename, codec='mp3')

    # Optional: remove the temporary file
    os.remove(audio_file)

    return mp3_filename

def extract_timestamps_and_text(input_text):
    lines = input_text.split("\n")
    timestamps = []
    text_without_timestamps = []

    for line in lines:
        if "-->" in line:
            timestamps.append(line)
        elif not line.isdigit() and not line.strip() == '':
            text_without_timestamps.append(line)

    first_timestamp = timestamps[0].split(" --> ")[0]
    last_timestamp = timestamps[-1].split(" --> ")[1]
    combined_text = " ".join(text_without_timestamps)

    return f"{first_timestamp}---> {last_timestamp} {combined_text}"

def prompt_factgpt_agent(user_input,pdf_names,web_urls,summary,video_description):
    _prompt = f""""You are a custom Agent designed to respond to user inputs truthfully and expertly to the best of your knowledge. Today's date is 1st February 2024. Your knowledge cutoff date is January 2023.

In order to help you answer the questions posed by the user, you are being provided access to a set of Tools. These tools include access to specific text sources and description of Youtube video. If you feel you are unable to confidently respond to the user's query based on your existing knowledge, you should determine whether the query can be better answered by consulting either the text sources or the database or the youtube video.

Tools:
1. Text Sources
   - PDF Names: {pdf_names}
   - Web URLs: {web_urls}
   - Summary: {summary}
2. YouTube Video:
   - Video Description: {video_description}

Your task is to analyze the user's query to decide whether it should be addressed using the text sources or the database or the you tube video.

If you determine the query is best answered using the Text Sources, your response should follow this JSON format:
[Strict JSON response format]:
{{
  "query": "User query as it is",
  "mode": "text"
}}


If you determine the query is best answered using the YouTube Video, your response should follow this JSON format:
[Strict JSON response format]:
{{
  "query": "User query as it is",
  "mode": "YouTube"
}}


Query Analysis:
- Examine the user's query to identify its nature and requirements.
- Decide if the query is seeking information that would be best sourced from the Text Sources or the Database provided.

Note: **DO NOT HALLUCINATE**. Choose the appropriate source based on the nature of the query. As an agent, your task is to provide accurate information based on the available tools.

User Input: {user_input}

**Response Format:**
{{
  "query": "User input as it is",
  "mode": "text/YouTube"
}}

"""
    return _prompt


def generate_answer(prompt,openai_params):
  message = [{"role":"user","content":prompt}]
  response = openai.ChatCompletion.create(messages=message,**openai_params)

  return response.choices[0].message.content


def tokenize_text_gpt(content,chunk_size=120,splitter_pattern=""):
    """
    Tokenize the text according to openai tokenizer using Langchain
    :param content:
    :return:
    """
    if not splitter_pattern:

        if "\n\n" in content:

            text_splitter_ = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=0,encoding_name="cl100k_base")
        elif "\n" in content:
            text_splitter_ = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=0,
                                                                         separator="\n",encoding_name="cl100k_base")
        else:
            text_splitter_ = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=0,
                                                                         separator=" ",encoding_name="cl100k_base")
    else:
        text_splitter_ = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size,chunk_overlap=0,
                                                                    separator=splitter_pattern,encoding_name="cl100k_base")
    passages = text_splitter_.split_text(content)

    return passages
import re
def get_embedding_list(texts, model="text-embedding-ada-002"):
  texts = [re.sub("\n+", " ", text) for text in texts]
  embedding_data = openai.Embedding.create(input = texts, model=model)['data']
  print("embeddings returned from openai")
  return [embedding_data[i]["embedding"] for i in range(len(embedding_data))]



def get_embedding(text, model="text-embedding-ada-002"):
  text = re.sub("\n+", " ", text)
  return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def create_index(text_chunks):

    all_chunks = []
    id_to_pdf_and_chunk = {}
    current_id = 0

    for pdf_name, chunks in text_chunks.items():
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            id_to_pdf_and_chunk[current_id] = (pdf_name, i)
            current_id += 1

    embeddings = get_embedding_list(all_chunks)
    if len(embeddings) == 0:
        print("No embeddings generated.")
        return None, {}

    dimension = len(embeddings[0])  # Dynamically get the dimension of embeddings

    index = hnswlib.Index(space='l2', dim=dimension)
    index.init_index(max_elements=len(all_chunks), ef_construction=200, M=16)

    # Bulk adding to the index
    index.add_items(embeddings)

    index.set_ef(50)
    return index, id_to_pdf_and_chunk

def search_similar_text(query, index, id_to_pdf_and_chunk, top_k):
    query_vector = get_embedding(query)
    try:
        labels, distances = index.knn_query(query_vector, k=top_k)

        # Flatten the labels and distances since we have a single query
        labels = labels.flatten()
        distances = distances.flatten()

        # Map labels back to PDF names and chunk indices
        results = [(id_to_pdf_and_chunk[label], distance) for label, distance in zip(labels, distances)]

    except Exception as e:
        print(e)
        results = []
    return results


def query2doc_prompt(query):
  _prompt = f"""Write a passage that answers the given query:

    Query: what state is this zip code 85282
    Passage: Welcome to TEMPE, AZ 85282. 85282 is a rural zip code in Tempe, Arizona. The population
    is primarily white, and mostly single. At $200,200 the average home value here is a bit higher than
    average for the Phoenix-Mesa-Scottsdale metro area, so this probably isn’t the place to look for housing
    bargains.5282 Zip code is located in the Mountain time zone at 33 degrees latitude (Fun Fact: this is the
    same latitude as Damascus, Syria!) and -112 degrees longitude.

    Query: why is gibbs model of reflection good
    Passage: In this reflection, I am going to use Gibbs (1988) Reflective Cycle. This model is a recognised
    framework for my reflection. Gibbs (1988) consists of six stages to complete one cycle which is able
    to improve my nursing practice continuously and learning from the experience for better practice in the
    future.n conclusion of my reflective assignment, I mention the model that I chose, Gibbs (1988) Reflective
    Cycle as my framework of my reflective. I state the reasons why I am choosing the model as well as some
    discussion on the important of doing reflection in nursing practice.

    Query: what does a thousand pardons means
    Passage: Oh, that’s all right, that’s all right, give us a rest; never mind about the direction, hang the
    direction - I beg pardon, I beg a thousand pardons, I am not well to-day; pay no attention when I soliloquize,
    it is an old habit, an old, bad habit, and hard to get rid of when one’s digestion is all disordered with eating
    food that was raised forever and ever before he was born; good land! a man can’t keep his functions
    regular on spring chickens thirteen hundred years old.

    Query: what is a macro warning
    Passage: Macro virus warning appears when no macros exist in the file in Word. When you open
    a Microsoft Word 2002 document or template, you may receive the following macro virus warning,
    even though the document or template does not contain macros: C:\<path>\<file name>contains macros.
    Macros may contain viruses.

    Query: {query}
    Passage:
 """
  return _prompt.strip()


def query_prompt_multi_file(Statement,filenames_with_descriptions):

    _prompt = f"""In this task, you are responsible for creating search queries that enable the efficient extraction of relevant information from specified files. Use the filenames and their short descriptions, as well as the instruction provided, to tailor your queries.
#### File List
{filenames_with_descriptions}
#### Instruction
{Statement}
#### Objectives
Your queries should aim to:
1. Identify the core theme or subject of the instruction within the file.
2. Pinpoint specific details, examples, or aspects that directly address the instruction.
3. Uncover any additional context or related information that could enrich the response to the instruction.

#### Note
Since you are aware of the filenames, there's no need to include them or the Statement in your queries or to query any external search engine.
Generate upto 10 queries. The queries should cover all the topics.

#### Your Queries
"""
    return _prompt


def extract_unique_chunks_from_pdf_chunks(pdf_chunks, fin_chunks):
  # Set to store unique chunks to avoid duplicates
    unique_chunks_set = set()
    
    for sublist in fin_chunks:
        for (pdf_name, chunk_index), _ in sublist:
            current_chunk = pdf_chunks[pdf_name][chunk_index]
            # Extract the text within <CHUNK>...</CHUNK>
            cleaned_chunk = current_chunk.split('<CHUNK> :')[-1].split('</CHUNK>')[0].strip()
            unique_chunks_set.add(cleaned_chunk)
    
    return list(unique_chunks_set)


def qa_prompt(instruction,context):
  _prompt = f"""When responding to instructions, ensure your answers are informed by specific details and key information from the provided document(s).
  Your response should:
  1. Be directly relevant to the task at hand, synthesizing essential details from the document(s) to create a cohesive answer.
  2. Critically assess and address any inconsistencies or inaccuracies between the instruction and document content.
  3. Focus on critical information, omitting any irrelevant details to maintain clarity and pertinence.
  4. Identify any content gaps and suggest alternatives or additional sources if necessary.
  5. Adhere to ethical standards, ensuring all information provided is accurate and based on the document(s) content.
  6. Provide a summary of key document insights as they relate to the specific instruction given.
  Your response should directly align with the nature of the instruction. **ALWAYS** Pick the relevant Chunks from the Context
  Context:{context}
  Instruction:{instruction} """

  return _prompt

def audio_video_ans(full_transcript,context_file,user_question):
  prompt_2 = f"""Your task involves analyzing an SRT (SubRip Text) file of a YouTube video, which includes text along with timestamps, and a context file outlining various contexts within the video. When presented with a user's question, you will:

  **Identify the Applicable Context**: Match the user's question to the most relevant context listed in the context file.
  **Locate the Corresponding Segment in the SRT File**: Find the section in the SRT file that aligns with the identified context, using the timestamps as a guide.
  **Formulate a Response**: Provide an answer based on the information found in the specific segment of the SRT file.
  **Include Context Timestamps in Your Answer**: Clearly state the start and end timestamps of the context, as detailed in the context file, which your answer is based on.
  Response Format:

  Answer Context: [Context Number and Timestamps]
  Response: [Your answer, clearly linked to the context and information in the transcript]

  Input:

  Full Transcript: {full_transcript}
  Context File: {context_file}
  User Question: {user_question}
  Begin Your Analysis: """
  return prompt_2

def audio_video_context(srt_content):
  _prompt = f"""Your task is to examine the provided SRT (SubRip Text) file, which contains both text and timestamps from a video. Identify the different contexts, or distinct topics and themes, present in the file. For each identified context:

  Analyze the text alongside its corresponding timestamps.
  Determine where a new context or topic begins and ends, noting the start and end timestamps.
  Briefly describe the main subject or theme of each context.
  List each context with its start and end timestamps as metadata.
  This analysis will categorize the video's content into distinct segments, each defined by its unique context and marked by precise timestamps, providing a clear breakdown of the video’s structure and content.

  SRT File for Analysis:
  {srt_content}

  Format for Response:
  Context [Number]: [Start_Timestamp] -- [End_Timestamp], [Description]

  Example: Context 1: 00:12:09,580 -- 00:12:11,940, [Description]
  ...
  Response: """

  return _prompt