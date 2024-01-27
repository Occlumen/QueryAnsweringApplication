from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, AnswerParser, PromptModel, PromptNode, PromptTemplate
from haystack.document_stores.weaviate import WeaviateDocumentStore
from haystack import Pipeline
from haystack import Document

#from haystack.components.converters import PyPDFToDocument
from model_add import LlamaCPPInvocationLayer

from fastapi import FastAPI, Depends,HTTPException,status,Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

import uvicorn 
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = "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"
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#print(OPENAI_API_KEY)
print("Import Successfully")

app = FastAPI()

frontend = Jinja2Templates(directory = "frontend")

def get_result(query):
    document_store = WeaviateDocumentStore(
        host="http://localhost",
        port= 8080,
        embedding_dim=768
    )
    print("Weaviate Document store created")
    prompt_template = PromptTemplate(
        prompt = """Given the provided Documents, answer the Query. Make your answer detailed and long \n
         Query : {query} \n
         Documents : {join(documents)} 
         Answer : """,
         output_parser=AnswerParser()
    )
    print("Prompt Template : ", prompt_template)

    def initialize_model():
        return PromptModel(
            model_name_or_path="model/mixtral-8x7b-v0.1.Q4_K_M.gguf",
            invocation_layer_class=LlamaCPPInvocationLayer,
            use_gpu=False,
            max_length=1000
        )
    model = initialize_model()
    prompt_node = PromptNode(
        model_name_or_path=model,
        max_length=1000,
        default_prompt_template=prompt_template
    )
    print("Prompt Node : ", prompt_node)

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Retriever : ", retriever)
    
    query_pipeline = Pipeline()
    query_pipeline.add_node(
        component=retriever, name="Retriever", inputs=["Query"]
    )
    query_pipeline.add_node(
        component=prompt_node, name="PromptNode",inputs=["Retriever"]
    )
    print("Query Pipeline : ", query_pipeline)

    json_response = query_pipeline.run( query=query, params={ "Retriever" : { "top_k" : 5 }})

    print("Answer : ", json_response)

    answers = json_response['answers']
    for ans in answers:
        print(ans.answer)
        answer = ans.answer
        break
    print("Answer : ")
    print(answer)

    # sentences = re.split(r'(?<=[.!?])\s', answer)
    # complete_sentences = [sentence for sentence in sentences if re.search(r'[.!?]$', sentence)]
    # updated_answer = ' '.join(complete_sentences)

    # chatgpt_template = PromptTemplate(
    #     prompt = """Given the explanation , please use the Feyman Technique to explain in simple, intuitive way. \n
    #      Answer : """,
    #      output_parser=AnswerParser()
    # )

    # chatgpt_node = PromptNode(
    #     model_name_or_path="gpt-3.5-turbo",
    #     api_key=OPENAI_API_KEY,
    #     max_length=1000,
    #     default_prompt_template=chatgpt_template
    # )
    
    # query_pipeline.add_node(
    #     component=chatgpt_node, name="ChatGPTNode",inputs=["PromptNode"]
    # )

    # print("Query Pipeline : ", query_pipeline)

    # json_response = query_pipeline.run( query=query, params={ "Retriever" : { "top_k" : 5 }})

    # print("Answer : ", json_response)

    # cg_answers = json_response['answers']
    # for cg_ans in cg_answers:
    #     print(cg_ans.answer)
    #     cg_answer = cg_ans.answer
    #     break
    # print("Answer : ")
    # print(cg_answer)

    # cg_sentences = re.split(r'(?<=[.!?])\s', cg_answer)
    # cg_complete_sentences = [sentence for sentence in cg_sentences if re.search(r'[.!?]$', sentence)]
    # cg_updated_answer = ' '.join(cg_complete_sentences)

    documents = json_response['documents']
    document_info = []
    for document in documents:
        content = document.content
        document_info.append(content)
    chat_documents = " "
    for i, doc_content in enumerate(document_info):
        chat_documents+= f"Document {i+1} Content : "
        chat_documents+= doc_content
        chat_documents+= "\n"

    print("Chat GPT Answer : ", chat_documents)

    return answer, chat_documents

@app.get("/")
async def index(request : Request):
    return frontend.TemplateResponse("index.html", {"request" : request })    

@app.post("/get_answer")
async def get_answer(request : Request, question : str = Form(...)):
    print(question)
    answer, chat_documents = get_result(question)
    response_data = jsonable_encoder(
        json.dumps({
            "answer" : answer,
            "chat_documents" : chat_documents
        })
    )
    res = Response(response_data)
    return res



