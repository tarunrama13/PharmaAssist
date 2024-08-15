__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from operator import itemgetter
import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_community.llms import Anthropic

import anthropic

def load_embeddings():
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=embed_model_name)
    

def load_model():
    return Anthropic(model="claude-instant-1",anthropic_api_key="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")

embeddings = load_embeddings()

model=load_model()
# Load or create vector store
def load_retriever():
    metadata_field_info = [
    AttributeInfo(
        name="Part",
        description="Part of document",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="Rule",
        description="number",
        type="integer",
    ),
    AttributeInfo(
        name="RuleTitle",
        description="heading of rule",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="PartTitle",
        description="description of Part",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="Subrule",
        description="it is a chunk of the rule",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="Document Name",
        type="string or list[string]",
    ),
    ]
    document_content_description = "information about drugs rule 1945"

    examples=[
           (
                "Give me details of rule 23",
                {
                    "query": "",
                    "filter": "eq(\"Rule\", '23')",
                },
            ),
            (
                "Give me details of subrule 3 of rule 24A",
                {
                    "query": "",
                    "filter": "and(eq(\"Rule\", '24A'),eq(\"Subrule\",3))",
                },
            ),
               (
                   "What are Import licences",
                {
                    "query": "Import licences",
                    "filter": "",
                },
               ),
               (
                "What is rule 4",
                {
                    "query": "",
                    "filter": "eq(\"Rule\", '4')",
                },
            ),
            (
                "What is the extend to which these rules applies",
                {
                    "query": "the extend",
                    "filter": "",
                },
            ),
            (
                "Give me details of rule titled as Functions",
                {
                    "query": "title Functions",
                    "filter": "",
                },
            ),
            (
                "give me details of rule titled as Registration Certificate for import of drugs",
                {
                    "query": "title Registration Certificate for import of drugs",
                    "filter": "",
                },
            ),
            ]
    prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
    examples=examples,
    
    )

    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | model | output_parser
    

    vectorstore=Chroma(persist_directory="data.db",embedding_function=embeddings)

    retriever=SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={'k':10}
    )


    return retriever

retriever = load_retriever()

# Function to answer questions
async def answer_question(question):

    keywords=["hi","hello","good","assist","who"]
    if any(keyword in question.lower() for keyword in keywords):
        template=f"""you are Pharma Assist, an AI assistance created by Pharma Assist to provide drug rule 1945 India information.
        answer the given question {question}. also use given information about yourself for question related to you. """
    else:
        docs= retriever.invoke(question)
        if(len(docs)>0):
            context=""
            for i in docs:
                context+=f"{i.page_content}+\n" 
            template = f"""
            Answer the question by analyzing given context .
            summarize the rules and related subrules
            please do not mention you get this details from provided context in your response.
            Context: {context}

            Question: {question}
            """
        else:
            template = f"""
            Mention for the given question that we don't have any information for this question as data is not available in drug rule 1945.
            Or provide more details in prompt.
            Question: {question}
            """
    
    return model.invoke(template)
    
@cl.on_message
async def on_message(message: cl.Message):
    
    msg = cl.Message(content="")
    query=message.content
    response=await answer_question(query)
    print(response)
    await cl.Message(content=response).send()