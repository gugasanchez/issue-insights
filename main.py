# 📚 GitHub Issues Knowledge Base with RAG (LangChain + OpenAI + Qdrant)

# This script:
# 1. Pulls closed issues from a GitHub repo
# 2. Embeds them using OpenAI embeddings
# 3. Stores in a vector DB (Qdrant)
# 4. Sets up a retrieval-based QA (RAG) system via LangChain

import os
import requests
from langchain.document_loaders import GitHubIssuesLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPO_OWNER = "your-org-or-user"
REPO_NAME = "your-repo"
QDRANT_COLLECTION = "github_issues"

# --- STEP 1: Load Issues from GitHub ---
def fetch_closed_issues(owner, repo):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=closed&per_page=100"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    issues = response.json()
    return [
        {
            "title": issue["title"],
            "body": issue.get("body", ""),
            "url": issue["html_url"],
        }
        for issue in issues if "pull_request" not in issue  # skip PRs
    ]

# --- STEP 2: Embed Issues ---
def embed_issues(issues):
    docs = [
        f"Title: {issue['title']}\nBody: {issue['body']}\nURL: {issue['url']}"
        for issue in issues
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.create_documents(docs)
    return split_docs

# --- STEP 3: Initialize Qdrant and store embeddings ---
def store_embeddings(docs):
    qdrant = QdrantClient(path="./qdrant_data")  # local mode
    if QDRANT_COLLECTION not in [col.name for col in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    embeddings = OpenAIEmbeddings()
    vectordb = Qdrant.from_documents(docs, embeddings, qdrant_client=qdrant, collection_name=QDRANT_COLLECTION)
    return vectordb

# --- STEP 4: Setup RAG Chain ---
def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain

# --- MAIN FLOW ---
if __name__ == "__main__":
    print("🔄 Fetching issues from GitHub...")
    issues = fetch_closed_issues(REPO_OWNER, REPO_NAME)
    
    print(f"✅ Fetched {len(issues)} closed issues.")
    
    print("🔍 Splitting and embedding documents...")
    docs = embed_issues(issues)

    print("🧠 Storing embeddings in Qdrant...")
    vectordb = store_embeddings(docs)

    print("💬 Initializing RAG chain...")
    qa_chain = create_qa_chain(vectordb)

    # Test it!
    print("🤖 Ask a question (e.g. 'Como resolvemos o bug de autenticação 500?')")
    while True:
        question = input("Pergunta: ")
        result = qa_chain({"query": question})
        print("\n🧠 Resposta:")
        print(result["result"])
        print("\n📎 Fontes:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'N/A')}")
