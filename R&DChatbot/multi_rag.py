import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

class MultiRAG:
    def __init__(self, pdf_folder="docs", db_path="faiss_index_pdf"):
        self.pdf_folder = pdf_folder
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.build_or_load_vectorstore()

    def build_or_load_vectorstore(self):
        print("📁 Loading PDFs...")
        documents = []
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.pdf_folder, filename))
                documents.extend(loader.load())

        print(f"📄 {len(documents)} documents loaded. Splitting and embedding...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.db_path)
        print("✅ FAISS index built from PDF files.")

    def retrieve_relevant_context(self, query, k=20):
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
