import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the Flan-T5 model and tokenizer for question answering
@st.cache_resource
def load_llm_model():
    model_name = "llama3"
    model= Ollama(model=model_name)
    return model

model = load_llm_model()

# Initialize embeddings using LangChain's HuggingFaceEmbeddings
@st.cache_resource
def load_embeddings():
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=embed_model_name)

embeddings = load_embeddings()

# Load or create vector store
@st.cache_resource
def load_vector_store():
    persist_directory = "fbc.db"
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    

vectorstore = load_vector_store()

# Create a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

# Function to answer questions
def answer_question(question):
    relevant_docs = retriever.invoke(question)
    
    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )


    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know". do not mention According to the provided context in the answer

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | output_parser
    )
    answer=chain.invoke({"question": question})
        
    return answer

# Streamlit UI
st.title("Drugs and Cosmetics Rules, 1945")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about the Drugs and Cosmetics Rules, 1945.pdf?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if vectorstore is not None:
        response = answer_question(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please make sure the vector store is properly initialized before asking questions.")

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()