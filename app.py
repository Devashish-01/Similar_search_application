import streamlit as st 
import os 
import logging 
from dotenv import load_dotenv
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

#setup logging
logging.basicConfig(level = logging.DEBUG)

#load Environemnt Variables
load_dotenv()

#ensure COHERE_API_KEY is set properly
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY environment variable not set")

st.set_page_config(page_title = "Educate Kids" , page_icon=":robot:")
st.header("SIMILARITY MATCHING")

model_name = "embed-english-v3.0"

#user_agent = "my-app/1.0" # Replace with your agent if you Want
#embeddings = CohereEmbeddings(model = model_name , user_agent = user_agent)
embeddings = CohereEmbeddings(model = model_name)

loader = CSVLoader(file_path = "myData.csv" , csv_args={
    'delimiter' : ',',
    'quotechar' : '"',
    'fieldnames' : ['words']
})

data = loader.load()

logging.debug(f"loaded data : {data}")
print(f"loaded data : {data}")

#extract data from data and ensure they are valid
texts = [doc.page_content for doc in data]

#validate the databeing passed to the mebeddings
for text in texts:
    logging.debug(f"Document text  : {text}")

#initialize FAISS database
try:
    db = FAISS.from_documents(data , embeddings)
    st.write("FAISS database created successfully")
except ValueError as e:
    logging.error(f"Error occured : {e}")
    for text in texts:
        try:
            embedding = embeddings.embed_documents([text])
            logging.debug(f"Embeddings : {embedding}")
        except ValueError as ve:
            logging.error(f"Failed to embed document: {text} with error :{ve}")

def get_input():
    input_text  = st.text_input("you : " , key = "input")
    return input_text

user_input = get_input()
submit = st.button("Find similar Things")

if submit:
    docs = db.similarity_search(user_input)
    print(docs)
    st.subheader("Top Matches : ")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)


