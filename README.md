# 🧠 issue-insights

Turn closed GitHub issues into a searchable, AI-powered knowledge base using RAG (Retrieval-Augmented Generation).

## ✨ What it does

- Fetches closed issues from any GitHub repo
- Embeds them using OpenAI embeddings
- Stores in a local Qdrant vector database
- Lets you ask natural language questions like:
  > "How did we fix the token expiration bug?"

## 🧰 Stack

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI API](https://platform.openai.com/docs)
- [Qdrant](https://qdrant.tech/)
- [GitHub REST API](https://docs.github.com/en/rest)

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/gugasanchez/issue-insights.git
cd issue-insights
```

### 2. Set up your environment

Create a `.env` file:

```env
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your-openai-key
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

You’ll be prompted to enter natural language questions. The system will search indexed issues and respond based on real team history.

## 📦 Example

```
🔄 Fetching issues from GitHub...
✅ Fetched 42 closed issues.
🔍 Splitting and embedding documents...
🧠 Storing embeddings in Qdrant...
💬 Initializing RAG chain...
🤖 Ask a question:
```

## 📌 Notes

- Currently fetches the **latest 100 closed issues** from the GitHub repo
- Skips PRs
- You can modify chunking, retriever settings, or extend to include PRs/commits

## 🧪 Future ideas

- Support Jira + GitHub hybrid ingestion
- Deploy as a FastAPI endpoint
- Add feedback loop to improve answers

---

Made with 💡 by people tired of forgetting how they fixed that bug last month.
