

#from langchain_community.document_loaders import PyPDFLoader
#from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS


#class RAGPipeline:
   # def __init__(self, pdf_path="data/sample.pdf"):
      #  loader = PyPDFLoader(pdf_path)
      #  documents = loader.load()

     #   text_splitter = RecursiveCharacterTextSplitter(
     #       chunk_size=500,
     #       chunk_overlap=50,
     #  )
      #  chunks = text_splitter.split_documents(documents)
#
      #  embeddings = HuggingFaceEmbeddings(
     #       model_name="all-MiniLM-L6-v2"
      #  )

      #  self.vectorstore = FAISS.from_documents(chunks, embeddings)

   # def query(self, question: str, k: int = 5):
   #     retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
    #    docs = retriever.invoke(question)

       # return [doc.page_content for doc in docs]


from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


class RAGPipeline:
    def __init__(
        self,
        pdf_path: str = "data/sample.pdf",
        index_path: str = "storage/faiss_index",
    ):
        self.pdf_path = pdf_path
        self.index_path = index_path

        # Embedding model for retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Load existing FAISS index if present already, otherwise build and save it
        if Path(self.index_path).exists():
            self.vectorstore = self._load_vectorstore()
            print("Loaded existing FAISS index from disk.")
        else:
            self.vectorstore = self._build_vectorstore()
            self._save_vectorstore()
            print("Built and saved new FAISS index.")

        # Generation model for answer creation
        self.generator = pipeline(
            "text-generation",
            model="distilgpt2",
            max_new_tokens=120,
        )

    def _build_vectorstore(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return vectorstore

    def _save_vectorstore(self):
        self.vectorstore.save_local(self.index_path)

    def _load_vectorstore(self):
        return FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def query(self, question: str, k: int = 5):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)

        return [
            {
                "content": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "page_label": doc.metadata.get("page_label"),
                },
            }
            for doc in docs
        ]

    def generate_answer(self, question: str, k: int = 5):
        docs = self.query(question, k=k)
        context = "\n\n".join([doc["content"] for doc in docs])

        prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:"""

        result = self.generator(prompt)
        full_text = result[0]["generated_text"]

        # Try to strip the prompt back out if the model echoes it
        answer = full_text.replace(prompt, "").strip()
        if not answer:
            answer = full_text.strip()

        return {
            "answer": answer,
            "context": docs,
        }