# Importing Libraries
import warnings
import streamlit as st
import os
from langchain_community.llms import GooglePalm
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


#%%
# Load environment variables
load_dotenv(find_dotenv())



#%%
from langchain.agents.agent_toolkits import(
    
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
    
    )



#%%
# Set API KEY
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")



#%%
# Instantiate GooglePalm
try:
    llm = GooglePalm(google_api_key=google_api_key, temperature=0.9)
except NotImplementedError as e:
    print(f"Error: {e}")
    print("It looks like the GooglePalm class or method is deprecated. Please check the documentation for alternatives.")
#%%
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
loader = PyPDFLoader('QatarPaperJournal.pdf')
pages = loader.load_and_split()

store = Chroma.from_documents(pages,embedding_function, collection_name='QatarPaperJournal')


#%%
#%%
vectorstore_info = VectorStoreInfo(
    name="QatarPaperJournal",
    description = "A research paper of developing multigeneration system and its energy and exergy analysis",
    vectorstore = store)
#%%

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

#%%
agent_executor = create_vectorstore_agent(
    llm = llm,
    toolkit = toolkit,
    verbose = True)


#%%

# Streamlit input and response
prompt = st.text_input('Input your prompt here')

if prompt:
    #response = llm(prompt)
    response =  agent_executor.run(prompt)
    st.write(response)
    
    
    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)
        
        
    
