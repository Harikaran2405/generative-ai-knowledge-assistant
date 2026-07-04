RAG-based AI assistant that answers questions from documents using semantic search and a local LLM
Generative AI Knowledge Assistant
I built this to answer questions about any PDF document — you upload a file, ask something in plain English, and it finds the relevant part of the document and generates an answer. It's a RAG (Retrieval-Augmented Generation) project, and everything runs on free, local models, no OpenAI API or paid key needed.
How it works
When you upload a PDF, it gets broken into small overlapping chunks of text using LangChain's text splitter, so nothing important gets cut off awkwardly between pieces. Each chunk is then converted into a vector, basically a set of numbers that captures its meaning, using a sentence-transformer model.
When you ask a question, it's converted into a vector too, and FAISS finds the chunks whose meaning is closest to your question. Those chunks get passed to a small language model, Flan-T5, along with your question, and it generates an answer grounded in that context.
Stack
Python, Streamlit for the UI, LangChain for document processing, Sentence-Transformers and FAISS for semantic search, and Hugging Face Transformers, Flan-T5, for generating answers.
Why a local model instead of GPT or OpenAI
Mainly to keep it free and self-contained, no API keys, no rate limits, nothing breaking if a key expires. Flan-T5-base isn't as capable as GPT-4, but for grounded Q&A over a single document it holds up fine.
Running it
Install the requirements file with pip, then run the app using Streamlit. Upload a PDF, type a question, get an answer.
What I'd improve next
Support other file types, not just PDF. Swap in a bigger model for better answers. Add memory so it can handle follow-up questions instead of treating each one separately. Deploy it somewhere permanent instead of an ngrok tunnel.
Author: Harikaran S
