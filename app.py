from haystack.components.builders import PromptBuilder,AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.converters import PyPDFToDocument
from haystack import Pipeline

#the generator is a python file with code for Llama cpp class : This component provides an interface 
#to generate text using a quantized model (GGUF) using llama.cpp.
from generator import LlamaCppGenerator

from fastapi import FastAPI, Depends,Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

import uvicorn 
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()

print("Import Successfully")

#the following variable openai_ai_key contains the api key generated from your Open Ai account.You can save the key in .env file 
#in the assignment folder or as a system variable
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")

#creates a API; FastAPI is a web framework to do that
app = FastAPI()

#Jinja2Templates is a templating language ; creates html component which can be passed to a request like HTTP request
frontend = Jinja2Templates(directory = "frontend")

#Function takes question from frontend and returns two text outputs 
def get_result(query):
    path_doc = ["data/scrum_master_en.pdf"]

    #Initialize Document Store
    document_store = InMemoryDocumentStore()

    print("Document Store Initialized !! ")

    #Convert PDF to List of Documents
    converter = PyPDFToDocument()
    output = converter.run(sources=path_doc)
    docs = output["documents"]

    print("Docs : ", type(docs))

    print("PDF converted !!")

    #Clean the list of Documents : will remove white spaces , empty lines and more
    preprocessor = DocumentCleaner()
    preprocessed_docs = preprocessor.run(docs)
    print(type(preprocessed_docs))
    print("The Documents are cleaned !!")

    #Split the Documents into words , sentence or passage : done to get better embeddings
    splitter = DocumentSplitter(split_by="word", split_length=750, split_overlap=0)
    splitted_docs = splitter.run(preprocessed_docs["documents"])
    print(type(splitted_docs))
    print("Document splitted into passages  !!")

    #Create embeddings of the document you got, using appropriate model
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")  
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(splitted_docs["documents"])
    print(type(documents_with_embeddings))
    print("Embedding are created")

    #store embeddings into the In Memory Document Store, so that we can use them later while generating our answer
    document_store.write_documents(documents_with_embeddings["documents"])
    print("Document Store Updated.")
    
    #prompt/template for first pipeline
    template = "Given the provided Documents, answer the Query. Make your answer brief and concise \n Query : {{query}} \n Answer : "
    #first pipeline : similar to chains in long chain ; helps create a structured flow of data
    rag_pipeline = Pipeline()
    
    #will be used for embedding of the query 
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    #Initialize the LlamaCppGenerator with necessary parameters
    model_path = "model/mixtral-8x7b-v0.1.Q4_K_M.gguf"
    generator = LlamaCppGenerator(model=model_path, n_ctx=2048, n_batch=512, model_kwargs={"n_gpu_layers": 70}, 
                                  generation_kwargs={"max_tokens": 50 , "top_p":0.3 })
    
    #Individually adding different components / functions , we will use in the pipeline
    rag_pipeline.add_component(instance=text_embedder,name="text_embedder")
    rag_pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=document_store, top_k=1, scale_score= 0.7), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=template), name="prompt_builder")
    rag_pipeline.add_component(instance=generator, name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    
    #connecting inputs and outputs of different components to create a flow/ pipeline for generating answer
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")
    
    #running the pipeline and also provide any additional variable / inputs to be sent to components
    result = rag_pipeline.run(data = {
        "text_embedder": {"text": query},
        "answer_builder": {"query": query}
        })
    #formatting the result of pipeline one
    generated_answer = result["answer_builder"]["answers"][0]
    answer = generated_answer.documents[0].content[0:4094]
    print(type(generated_answer))
    
    #prompt/template for second pipeline
    chatgpt_template = "Given the explanation {{ answer }} , please use the Feyman Technique to explain in simple, intuitive way."
    #creating second pipeline
    pipe = Pipeline()

    #Individually adding different components / functions , we will use in the pipeline
    pipe.add_component("prompt_builder", PromptBuilder(template= chatgpt_template))
    pipe.add_component("llm", OpenAIGenerator(api_key=OPENAI_API_KEY))
    #connecting inputs and outputs of different components to create a flow/ pipeline for generating answer
    pipe.connect("prompt_builder", "llm")
    
    #running the pipeline and also provide any additional variable / inputs to be sent to components
    response=pipe.run({
                 "prompt_builder": {
                 "answer": answer } })

    print(response)
    #formatting the result of pipeline second
    chat_gpt_response = response["llm"]["replies"]
    
    #returning the formatted answer from both the pipelines
    return answer, chat_gpt_response

@app.get("/")
async def index(request : Request):
    return frontend.TemplateResponse({"request" : request }, "index.html",)    

@app.post("/get_answer")
async def get_answer(request : Request, question : str = Form(...)):
    print(question)
    #calling the function from above
    answer, chat_documents = get_result(question)
    #will send data to html file for frontend display
    response_data = jsonable_encoder(
        json.dumps({
            "answer" : answer,
            "chat_documents" : chat_documents
        })
    )
    res = Response(response_data)
    return res



