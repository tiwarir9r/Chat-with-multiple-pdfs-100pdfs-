import streamlit as st
import os
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from pytube import YouTube
from moviepy.editor import *
import openai
import json
from utils import *
# Load your OpenAI API key
openai.api_key = ""
openai_params1 = {"model":"gpt-3.5-turbo-1106",
                 "temperature":0.5,
                 "frequency_penalty":0.0,
                 "presence_penalty":0.0,
                 "max_tokens":1500,
                 "top_p":1}
# Function definitions (from the provided code)
# ... (include all the function definitions from the provided code)

def main():
    st.title("PDFGPT")

    # File uploads
    st.subheader("Upload Files")
    uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    uploaded_urls = st.text_area("Enter URLs (one per line)")

    if st.button("Process Files"):
        # Create the 'summaries' directory if it doesn't exist
        os.makedirs("summaries", exist_ok=True)

        # Process PDFs
        pdf_summaries = {}
        for pdf_path in uploaded_pdfs:
            file_path = os.path.join("temp", pdf_path.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(pdf_path.getvalue())

            text = extract_text_from_pdf(file_path)
            print(len(text))

            if count_tokens(text) > 16000:
                summary_txt = generate_summary(truncate_to_token_limit(text, 14000))
            else:
                summary_txt = generate_summary(text)

            pdf_summaries[pdf_path.name] = summary_txt
            print(summary_txt)

        # # Write all PDF summaries to a single file
        # with open("summaries/pdf_summary.txt", "w", encoding="utf-8") as f:
        #     for pdf_name, summary in pdf_summaries.items():
        #         f.write(f"PDF: {pdf_name}\n\n{summary}\n\n{'='*50}\n\n")

        # Process URLs
        url_summaries = {}
        for url in uploaded_urls.split("\n"):
            url = url.strip()
            if url:
                text = scrape_url(url)
                summary = generate_summary(text)
                pdf_summaries[url] = summary

        # Write all URL summaries to a single file
        with open("summaries/summary.txt", "w", encoding="utf-8") as f:
            json.dump(pdf_summaries, f, ensure_ascii=False, indent=4)

        # Create a single file for chunks (PDFs and URLs)
        
        with open("summaries/chunks.txt", "w", encoding="utf-8") as f:
            pdf_chunks = {}
            for pdf_path in uploaded_pdfs:
                file_path = os.path.join("temp", pdf_path.name)
                text = extract_text_from_pdf(file_path)
                chunks = tokenize_text_gpt(text, chunk_size=256)
                summary_txt = pdf_summaries[pdf_path.name]
                chunks_modf = []
                for chunk in chunks:
                    chunk_with_summary = f"<CHUNK> : {chunk} </CHUNK>\n\n<PDF SUMMARY> : {summary_txt} </PDF SUMMARY>"
                    chunks_modf.append(chunk_with_summary)

                pdf_chunks[pdf_path.name] = chunks_modf
            for url in uploaded_urls.split("\n"):
                url = url.strip()
                if url:
                    text = scrape_url(url)
                    chunks = tokenize_text_gpt(text, chunk_size=256)
                    summary = pdf_summaries[url]
                    chunks_modf = []
                    for chunk in chunks:
                        chunk_with_summary = f"<CHUNK> : {chunk} </CHUNK>\n\n<URL SUMMARY> : {summary} </URL SUMMARY>"
                        chunks_modf.append(chunk_with_summary)
                    pdf_chunks[url] = chunks_modf
            json.dump(pdf_chunks, f, ensure_ascii=False, indent=4)
        st.success("Files processed successfully!")

    # User query
    st.subheader("Ask a Question")
    user_query = st.text_area("Enter your query")

    if st.button("Get Answer"):
        if not user_query:
            st.warning("Please enter a query.")
            return

        chunk_file_path = "summaries/chunks.txt"
        summary_path  = "summaries/summary.txt"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_pdfs = json.load(f)
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            chunks_dict = json.load(f)
        print(type(chunks_dict))
        # Retrieve chunks from chunks.txt (Now directly using the content)
        index, id_to_pdf = create_index(chunks_dict)  # Assuming create_index can handle the content directly
        
        # Based on input query
        query = user_query  # Using the query input by the user
        query_modf = generate_answer(query2doc_prompt(query), openai_params)
        query_modf = "<QUERY>" + " : " + query + "</QUERY>" + "\n\n" + "<HYPOTHETICAL ANSWER>" + " : " + query_modf + "</HYPOTHETICAL ANSWER>"
        results = search_similar_text(query, index, id_to_pdf, 60)
        
        print("results:",results)
        # Further processing...
        pdf_counts = {}
        for result in results:
            pdf_name, chunk_index = result[0]
            pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
        k = 6
        pdf_names = [pdf_name for pdf_name, count in pdf_counts.items() if count > k]
        fin_pdfs = set(pdf_names)
        
        filenames_with_descriptions = {}
        documents = {}
        for i in fin_pdfs:
            modf_docs = []
            print(i)
            # print(summary_pdfs[i])
            modf_docs.extend(chunks_dict[i])
            documents[i] = modf_docs
            filenames_with_descriptions[i] = summary_pdfs[i]
        print(documents)          
        index1, id_to_pdf2 = create_index(documents)  # Re-index based on new documents
        
        prompt = query_prompt_multi_file(Statement=query, filenames_with_descriptions=filenames_with_descriptions)
        fin_chunks_qa_ans = []
        for i in generate_answer(prompt, openai_params1).split('\n'):
            results = search_similar_text(i, index1, id_to_pdf2, 20)
            fin_chunks_qa_ans.append(results)
        
        final_chunks = extract_unique_chunks_from_pdf_chunks(chunks_dict, fin_chunks_qa_ans)  # Adapt this to use content directly
        
        prompt = qa_prompt(instruction=query, context=final_chunks)
        answer = generate_answer(prompt, openai_params)
        st.write(answer)

if __name__ == "__main__":
    main()