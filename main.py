from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = 'data/sample.pdf'

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded pages : {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

print("\nFirst chunk preview:\n")
print(chunks[0].page_content[:1000])


# EMBEDDINGS + VECTOR STORE

#from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import FAISS
#import os 

#API key set up

#os.environ["OPEN_API_KEY"] = ""

#embeddings = openAIEmbeddings()
#vectorstroe = FAISS.from_documents(chunks,embeddings)

# using local embeddings --> sentence-transformers

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os 

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks,embeddings)
print("Vector store created!")

#Retrieval

query = "What is the Transformer architecture?"

retriever = vectorstore.as_retriever(search_kwargs={"k":5})
retrieved_docs = retriever.invoke(query)

print("\nTop retrieved chunks:\n")

for i, doc in enumerate(retrieved_docs,start=1):
	print(f"\n Chunk{i}\n")
	print("Metadata:",doc.metadata)
	print(doc.page_content[:700])

#Generate answers from retrieved chunks

from transformers import pipeline

context = "\n".join([doc.page_content for doc in retrieved_docs])

prompt = f""" Answer the question based only on the context below.
Context:{context}
Question:{query}
Answer:"""

generator = pipeline("text-generation", model ="google/flan-t5-base", max_new_tokens=150)

result = generator(prompt)

print("\nFinal answer\n")
print(result[0]["generated_text"])