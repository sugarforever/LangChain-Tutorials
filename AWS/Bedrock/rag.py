from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from langchain.llms import Bedrock

retriever = AmazonKnowledgeBasesRetriever(
    credentials_profile_name="william",
    knowledge_base_id="GFQSZ3PZJV",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

question = "Introduce the training hardware of llama2"
docs = retriever.get_relevant_documents(query=question)
print(docs)

print("\n******************************\n")

llm = Bedrock(
    credentials_profile_name="william",
    model_id="amazon.titan-text-express-v1"
)

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

response = qa(question)
print(response)
