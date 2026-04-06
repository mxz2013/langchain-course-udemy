import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama

from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()

# note that in order to make the search work, we should use exactly the same embedding model here for the query as we used for the documents when we created the vector store. Otherwise, the search results will be poor.

print("Initializing components ...")

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)  # create an instance of the OpenAIEmbeddings class

# llm = ChatOpenAI()
llm = ChatOllama(model="gpt-oss:latest")  # create an instance of the OllamaChat class

vectorsotre = PineconeVectorStore(
    embedding=embeddings, index_name=os.environ.get("INDEX_NAME")
)

retriever = vectorsotre.as_retriever(
    search_kwargs={"k": 3}
)  # create a retriever from the vector store, return k most similar documents sortted by similarity score


prompt_template = ChatPromptTemplate.from_template(
    """Answer the quesitn based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string

    Args:
        docs (_type_): _description_

    Returns:
        _type_: _description_
    """
    return "\n\n".join([doc.page_content for doc in docs])


# implementation 1: without LCEL (Simple Function-Based Approach)
def retrieval_chain_without_lcel(query: str):
    """A simple retrieval chain without using LCE
    Manually retrieves documents, formats them, and then constructs a prompt to send to the LLM.
    Limitations:
    - Manual steep by step execution
    - No build-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More vebose and error-prone due to manual handling of each step

    Args:
        query (_type_): _description_

    Returns:
        _type_: _description_
    """

    # step 1: retrieve relevant documents
    docs = retriever.invoke(query)

    # step 2: format the retrieved documents into a single string
    context = format_docs(docs)

    # step 3: construct the prompt with the formatted context and the original query
    messages = prompt_template.format(context=context, question=query)  # send to LLM

    # step 4: invoke the LLM with the constructed prompt
    response = llm.invoke(messages)

    # step 5: return the LLM's response
    return response.content


# ========================================================================
# Option 2: Use implementation WITH LCEL (Better Approach)
# ========================================================================
def create_retrieval_chain_with_lcel():
    """Create a retrieval chain using LCEL
    Returns a chain that can be invoked with {"question:" "..."}
    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("retrieving documents ...")
    # query
    query = "what is Pinecone in maching learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
