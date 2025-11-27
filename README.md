# 🔐 Local Air-Gapped AI Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A fully offline AI agent for secure, air-gapped environments. Built with **LangChain**, **llama.cpp**, and **FAISS** for local document Q&A without cloud dependencies.

## ✨ Features

- 🔒 **100% Offline** - No internet or cloud API calls required
- 🤖 **Local LLM** - Run models like Phi-3, Llama, Mistral via llama.cpp
- 📚 **Document Q&A** - Query PDFs, DOCX, and TXT files locally
- 🔍 **Vector Search** - FAISS-powered semantic search
- 🎯 **Air-Gap Ready** - Perfect for IL5, classified, or high-security environments
- ⚡ **CPU Optimized** - Works without GPU (GPU acceleration optional)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local AI Agent                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  LLM Module  │  │  Embeddings  │  │ Vector Store │     │
│  │  (llama.cpp) │  │(Sentence-TR) │  │   (FAISS)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                          │                                  │
│                   ┌──────▼──────┐                          │
│                   │    Agent     │                          │
│                   │  (LangChain) │                          │
│                   └──────────────┘                          │
│                          │                                  │
│                   ┌──────▼──────┐                          │
│                   │   CLI/API    │                          │
│                   └──────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

         Data Flow: Documents → Embeddings → Vector DB → Query → LLM → Answer
```

## 📁 Project Structure

```
local-airgapped-agent/
│
├── app/                          # Core application modules
│   ├── offline_llm.py            # LLM interface (llama.cpp)
│   ├── embeddings.py             # Document processing & embeddings
│   ├── vector_store.py           # FAISS vector database
│   └── agent.py                  # LangChain agent logic
│
├── models/                       # Place GGUF models here
│   └── README.md                 # Model download instructions
│
├── data/                         # Your documents
│   └── sample_docs/              # Example documents
│
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- 5GB+ free disk space (for models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/surajmaraboina/local-airgapped-agent.git
   cd local-airgapped-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a model** (GGUF format)

   Choose one:
   
   - **Phi-3 Mini** (2.7GB) - Recommended for CPU
     ```bash
     mkdir -p models
     cd models
     # Download from HuggingFace
     wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
     ```

   - **Llama 3.1 8B** (4.7GB)
     ```bash
     # Download from HuggingFace
     wget https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
     ```

4. **Add your documents**
   ```bash
   mkdir -p data
   # Place your PDF, DOCX, or TXT files in the data/ folder
   ```

### Usage

#### Option 1: Command Line (Coming Soon)

```bash
python main.py --model models/Phi-3-mini-4k-instruct-q4.gguf --data data/
```

#### Option 2: Python API

```python
from app.offline_llm import create_offline_llm
from app.embeddings import create_embedder
from app.vector_store import VectorStore

# Initialize components
llm = create_offline_llm("models/Phi-3-mini-4k-instruct-q4.gguf")
embedder = create_embedder()

# Process documents
docs = embedder.process_document("data/report.pdf")
embedded_docs = embedder.embed_documents([docs])

# Build vector store
vector_store = VectorStore(embedding_dim=embedder.get_embedding_dimension())
vector_store.add_documents(embedded_docs)

# Query
query = "What are the key findings?"
results = vector_store.search(embedder.embed_text(query), k=3)

# Generate answer
context = "\n".join([r['text'] for r in results])
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
answer = llm.generate(prompt)
print(answer)
```

## 🎯 Use Cases

### Perfect For:

- 🏛️ **Government/Defense** - IL5, classified environments
- 🏥 **Healthcare** - HIPAA-compliant document analysis
- 🏦 **Finance** - Secure internal knowledge base
- 🔬 **Research** - Air-gapped lab environments
- 🛡️ **Enterprise** - High-security corporate networks

### Example Queries:

```
"Summarize the security policy document"
"What are the compliance requirements mentioned?"
"Extract key financial figures from the report"
"Find all references to Project X"
```

## ⚙️ Configuration

### LLM Settings

Edit `app/offline_llm.py` or pass parameters:

```python
llm = create_offline_llm(
    model_path="models/your-model.gguf",
    n_ctx=2048,           # Context window
    n_gpu_layers=0,       # 0 for CPU-only, 35+ for GPU
    temperature=0.7,      # Generation temperature
    max_tokens=512        # Max output length
)
```

### Embedding Model

Change in `app/embeddings.py`:

```python
embedder = create_embedder(
    model_name="all-MiniLM-L6-v2",  # Fast, good quality
    # model_name="all-mpnet-base-v2",  # Better quality, slower
    device="cpu"  # or "cuda" for GPU
)
```

## 📊 Performance

| Model | Size | RAM | Speed (tokens/s) | Quality |
|-------|------|-----|------------------|----------|
| Phi-3 Mini Q4 | 2.7GB | 4GB | 15-30 | Good |
| Llama 3.1 8B Q4 | 4.7GB | 8GB | 10-20 | Better |
| Mistral 7B Q4 | 4.1GB | 6GB | 12-25 | Better |

*Benchmarked on AMD Ryzen 9 5900X (12 cores)*

## 🔧 Troubleshooting

### Common Issues:

1. **"Model not found" error**
   - Ensure GGUF model is in `models/` directory
   - Check file path is correct

2. **Slow generation**
   - Use Q4 quantized models (smaller, faster)
   - Reduce `n_ctx` parameter
   - Enable GPU acceleration if available

3. **Out of memory**
   - Use smaller model (Phi-3 Mini)
   - Close other applications
   - Reduce batch size

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Fast LLM inference
- [LangChain](https://github.com/langchain-ai/langchain) - Agent framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [sentence-transformers](https://www.sbert.net/) - Embeddings

## ⭐ Star History

If this project helped you, please star it on GitHub!

## 📞 Contact

For questions or issues, please [open an issue](https://github.com/surajmaraboina/local-airgapped-agent/issues).

---

**Built for Frontier Foundry & secure AI deployments** 🚀
