from haystack.nodes import AnswerParser , EmbeddingRetriever , PreProcessor
from haystack.document_stores.weaviate import WeaviateDocumentStore
from haystack.nodes import PDFToTextConverter
#from haystack.components.converters import PyPDFToDocument

print("Import Successful !! ")


path_doc = ["data/scrum_master_en.pdf"]

#Initialize Document Store

document_store = WeaviateDocumentStore(
    host = "http://localhost",
    port = 8080 ,
    embedding_dim =768
)

print("Document Store Initialized !! ")

converter = PDFToTextConverter()
#output = converter.run(paths=path_doc)
docs = converter.convert(file_path=path_doc)

print("Docs : ", docs)

final_doc = []

for doc in docs :
    print(doc.text)
    new_doc = {
        'content' : doc.text ,
        'meta' : doc.metadata
    }
    final_doc.append(new_doc)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="sentence",
    split_length=1000,
    split_respect_sentence_boundary=True
)

preprocessed_docs = preprocessor.process(final_doc)

document_store.write_documents(preprocessed_docs)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Retriever Initialized !! ")

document_store.update_embeddings(retriever)

print("Embeddings Done.")
    