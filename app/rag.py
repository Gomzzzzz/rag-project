

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


from pathlib import Path  # for checking whether the saved FAISS index already exists

from langchain_community.document_loaders import PyPDFLoader  # loads text from PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into chunks
from langchain_community.embeddings import HuggingFaceEmbeddings  # converts text into embeddings
from langchain_community.vectorstores import FAISS  # vector database for similarity search
from transformers import pipeline  # Hugging Face generation pipeline


class RAGPipeline:
    def __init__(
        self,
        pdf_path: str = "data/sample.pdf",
        index_path: str = "storage/faiss_index",
    ):
        self.pdf_path = pdf_path  # path to the source PDF
        self.index_path = index_path  # path where FAISS index will be stored

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )  # embedding model used only for retrieval, not for answer generation

        if Path(self.index_path).exists():
            self.vectorstore = self._load_vectorstore()  # reuse saved index if already built
            print("Loaded existing FAISS index from disk.")
        else:
            self.vectorstore = self._build_vectorstore()  # build a new index from the PDF
            self._save_vectorstore()  # save index so startup is faster next time
            print("Built and saved new FAISS index.")

        self.generator = pipeline(
            "text-generation",
            model="distilgpt2",
        )  # weak local generation model, but enough to complete pipeline

    def _build_vectorstore(self):
        loader = PyPDFLoader(self.pdf_path)  # initialize PDF loader
        documents = loader.load()  # load PDF pages as Document objects

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )  # split long text into overlapping chunks for better retrieval

        chunks = text_splitter.split_documents(documents)  # convert pages into smaller retrievable chunks

        vectorstore = FAISS.from_documents(chunks, self.embeddings)  # embed chunks and store them in FAISS
        return vectorstore

    def _save_vectorstore(self):
        self.vectorstore.save_local(self.index_path)  # persist FAISS index to local disk

    def _load_vectorstore(self):
        return FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )  # load saved FAISS index using the same embedding model

    def query(self, question: str, k: int = 5):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )  # create retriever that returns top-k similar chunks

        docs = retriever.invoke(question)  # embed query and retrieve relevant chunks

        return [
            {
                "content": doc.page_content,  # actual retrieved chunk text
                "metadata": {
                    "source": doc.metadata.get("source"),  # source file path
                    "page": doc.metadata.get("page"),  # zero-based page index
                    "page_label": doc.metadata.get("page_label"),  # human-readable page number
                },
            }
            for doc in docs
        ]  # return structured retrieval results

    def generate_answer(self, question: str, k: int = 3):
        docs = self.query(question, k=k)  # first retrieve relevant chunks

        context_chunks = [
            doc["content"].strip()
            for doc in docs
            if doc["content"] and doc["content"].strip()
        ]  # keep only non-empty chunk text

        context = "\n\n".join(context_chunks[:3])  # keep context short so weak model gets less confused

        prompt = f"""You are answering based only on the provided context.
Give a short, direct answer in 2 to 4 sentences.
Do not invent information.
If the answer is not in the context, say: "I could not find the answer in the provided context."

Context:
{context}

Question: {question}

Answer:"""  # prompt tells the model to stay grounded and concise

        result = self.generator(
            prompt,
            max_new_tokens=80,
            do_sample=False,
        )  # deterministic generation with a short output limit

        full_text = result[0]["generated_text"]  # raw generated text from model

        answer = full_text.replace(prompt, "").strip()  # remove echoed prompt if model repeats it

        if not answer:
            answer = full_text.strip()  # fallback in case prompt removal leaves empty text

        answer_lines = [line.strip() for line in answer.splitlines() if line.strip()]  # remove blank/noisy lines
        answer = " ".join(answer_lines)  # flatten answer into one clean string

        if len(answer) < 20:
            answer = "I could not generate a reliable answer from the retrieved context."  # fallback for weak output

        return {
            "answer": answer,
            "context": docs,
        }  # final answer plus supporting retrieved chunks