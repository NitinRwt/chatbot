
# 💼 Multi-Task Gemini Chatbot for Work Platforms

This project implements an end-to-end AI chatbot using **Google Gemini API** that serves three roles in a unified interface:

1. **Onboarding Professionals**
2. **Searching Work Opportunities**
3. **Creating Work Opportunities (by Clients)**

It uses **intent detection**, **retrieval-augmented generation (RAG)** via FAISS, and few-shot prompting to enable rich and structured outputs.

---

## 🚀 Features

- 🌐 **Unified Chatbot** powered by Gemini Pro
- 🔎 **RAG (Retrieval-Augmented Generation)** for profile and job retrieval
- 🧠 **Few-shot Intent Detection** to classify user tasks
- 📂 **Modular & Extensible** Python codebase
- ⚡ **Fast Local Search** with FAISS vector database
- 🧾 Converts natural language prompts into structured job posts (JSON)

---

## 🧱 Project Structure

```

multi\_task\_gemini/
├── data/
│   ├── onboarding\_profiles.jsonl       # Sample user profiles
│   ├── job\_listings.jsonl              # Available job postings
│   ├── client\_prompts.jsonl            # Prompts → job post mappings
├── faiss\_index/
│   ├── profiles\_index.bin              # FAISS vector index for profiles
│   ├── jobs\_index.bin                  # FAISS vector index for jobs
│   ├── profile\_id\_map.json             # Maps FAISS vector IDs to profile JSON
│   └── job\_id\_map.json                 # Maps FAISS vector IDs to job JSON
├── generate\_embeddings.py              # Embeds text from datasets
├── build\_faiss\_index.py                # Builds and saves FAISS index
├── rag\_utils.py                        # Helper functions for retrieval
├── intent\_detector.py                  # Few-shot intent detection
├── multi\_task\_gemini.py                # Main chatbot logic
├── requirements.txt                    # Python dependencies
└── README.md                           # You’re here!

````

---

## 📦 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/NitinRwt/chatbot.git
cd chatbot
````

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python3 -m venv env
source env/bin/activate     # Windows: .\env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Gemini API Key Setup

1. Go to [Google AI Studio](https://makersuite.google.com).
2. Generate an API key under “API Keys.”
3. Export your API key:

```bash
# Linux/macOS
export GOOGLE_API_KEY="your-api-key"

# Windows PowerShell
$env:GOOGLE_API_KEY="your-api-key"
```

---

## 🧠 Data Preparation

All datasets are in **JSON Lines** format (`.jsonl`). Examples:

### 📘 `data/onboarding_profiles.jsonl`

```json
{"id": "user_001", "name": "Alice Johnson", "role": "Backend Developer", ...}
```

### 📘 `data/job_listings.jsonl`

```json
{"id": "job_002", "title": "Frontend Engineer", "skills": ["React", "HTML"], ...}
```

### 📘 `data/client_prompts.jsonl`

```json
{"prompt": "We need a data analyst with Python and Tableau", "output": {"title": ..., "skills": [...], ...}}
```

---

## 🛠️ Build Embeddings & Indexes

### 1. Generate Embeddings for Job Listings & Profiles

```bash
python generate_embeddings.py
```

### 2. Build FAISS Indexes

```bash
python build_faiss_index.py
```

---

## 💬 Run the Chatbot

```bash
python multi_task_gemini.py
```

You’ll be able to type natural queries like:

* ✅ “I’m a React developer with 3 years experience, looking for remote jobs.”
* 🆕 “I want to sign up. My name is Rahul, I’m a frontend dev in Bangalore.”
* 💼 “We’re hiring a Django developer for a short-term project.”

---

## 🔍 Testing Examples

| Use Case             | Input Prompt                                                |
| -------------------- | ----------------------------------------------------------- |
| Onboard Professional | “Hi, I’m Aditi, a backend dev with 2 years exp in Django.”  |
| Search Work          | “Any remote React jobs for me?”                             |
| Create Job           | “I need a fullstack engineer with React and Django skills.” |

---

## 🌐 Optional: Run as a Web App (Flask)

You can later integrate this prototype into a Flask API or Gradio UI.

---

## 🔮 Roadmap

* [ ✔️] Web UI using Gradio
* [ ✔️] MongoDB for persistent storage
* [ ✔️] Pinecone or Weaviate for production vector storage
* [ ✔️] Chat memory via langchain-like architecture

