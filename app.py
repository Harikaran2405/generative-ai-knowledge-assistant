import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile

st.set_page_config(page_title="Generative AI Knowledge Assistant", page_icon="📄")
st.title("📄 Generative AI Knowledge Assistant")
st.write("Upload a document and ask questions about it — answered using semantic search + a local AI model.")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

tokenizer, model = load_model()
embeddings = load_embeddings()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success(f"Document processed into {len(chunks)} chunks. Ask away!")

    question = st.text_input("Ask a question about the document:")

    if question:
        relevant_chunks = vectorstore.similarity_search(question, k=3)
        context = "\n".join([c.page_content for c in relevant_chunks])

        prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question: {question}
Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Answer:")
        st.write(answer)

        with st.expander("See retrieved context"):
            st.write(context)
