from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.chroma import ChromaTranslator
from huggingface_hub import login
login(token ="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")
import torch
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_community.llms.ollama import Ollama

embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # A good balance of speed and performance
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)


template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
output_parser = StrOutputParser()


model = Ollama(model="llama3")
vectorstoreFBC=Chroma(persist_directory="fbc.db",embedding_function=embeddings)
retriever=vectorstoreFBC.as_retriever()

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | output_parser
)
chain.invoke({"question": "what are import licenses'?"})
