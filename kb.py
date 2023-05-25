from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex, 
    LangchainEmbedding, 
    LLMPredictor, 
    ServiceContext, 
    StorageContext, 
    download_loader,
    PromptHelper,
    load_index_from_storage
)
import gradio as gr
import sys
import os


def data_ingestion_indexing(directory_path):

    #constraint parameters
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    #allows the user to explicitly set certain constraint parameters
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    local_llm_path = './ggml-gpt4all-j-v1.3-groovy.bin' # not in this repo, but in local directory only due to large file size
    
    #LLMPredictor is a wrapper class around LangChain's LLMChain that allows easy integration into LlamaIndex
    llm_predictor = LLMPredictor(llm=GPT4All(model=local_llm_path, backend='gptj', streaming=True, n_ctx=512))
    
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

    prompt_helper = PromptHelper(max_input_size=512, num_output=256, max_chunk_overlap=-1000)
    
    #constructs service_context
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=embed_model,
        prompt_helper=prompt_helper,
        node_parser=SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=300, chunk_overlap=20))
    )

    #loads data from the specified directory path
    documents = SimpleDirectoryReader(directory_path).load_data()

    #when first building the index
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    #persist index to disk, default "storage" folder
    index.storage_context.persist(persist_dir="./storage")

    return index

def data_querying(input_text):

    #rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    #loads index from storage
    index = load_index_from_storage(storage_context)
    
    #queries the index with the input text
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=1, service_context=service_context)
    response = query_engine.query(input_text)
    
    return response.response

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your question"),
                     outputs="text",
                     title="Wenqi's Custom-trained DevSecOps Knowledge Base")

#passes in data directory
index = data_ingestion_indexing("data")
iface.launch(share=False)
