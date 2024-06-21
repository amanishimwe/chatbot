from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import json

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
with open(os.path.join(parent_dir, 'config.json')) as f:
    config = json.load(f)

API_KEY = config.get('API_KEY') 
# Initialize the Sentence Transformer Embeddings model
embeddings_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# Set up the Chroma vector database
vector_db = Chroma(
    persist_directory='../vector_db',  # Ensure this directory exists and is writable
    collection_name='census2014',
    embedding_function=embeddings_model
)

# Pull a prompt template from LangChain Hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Configure the retriever based on the vector database
retriever = vector_db.as_retriever()

# Initialize the ChatOpenAI model with your OpenAI API key
llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0.5)

# Set up conversation memory
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Create a document combination chain
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# Create the retrieval question answering chain
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Define a function to handle user questions and return responses
def rag_func(question: str) -> str:
    """This function takes a user question and returns a response."""
    response = qa_chain.invoke({"input": question,})
    return response["answer"]
