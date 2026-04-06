# this ingestion script is responsible for loading the text file, splitting it into chunks, creating embeddings, and storing them in a vector store (Pinecone) for later retrieval.
# we only need to run this script once to ingest the data, and then we can use the vector store for retrieval in our main application.
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


if __name__ == "__main__":
    print("Ingesting...")
    # Load the text file
    loader = TextLoader(
        "mediumblog1.txt"
    )  # , encoding="utf-8", autodetect_encoding=True)
    documents = loader.load()  # load the fiele to langchain document format
    print("splinting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)  # split the document into chunks
    print(f"created {len(text)} chunks")
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )  # create an instance of the OpenAIEmbeddings class

    print("creating vector store...")

    PineconeVectorStore.from_documents(
        documents=text,
        embedding=embeddings,
        index_name=os.environ.get("INDEX_NAME"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
    )  # create a vector store from the documents and embeddings
    print("ingestion complete!")
