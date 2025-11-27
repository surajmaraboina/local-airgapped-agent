#!/usr/bin/env python3
"""🧪 FUNCTIONAL TEST - Verifies Core System Works!"""
print("="*60)
print("🚀 LOCAL AI AGENT - FUNCTIONAL TEST")
print("="*60 + "\n")

print("1️⃣  Testing imports...")
try:
    from app.embeddings import create_embedder
    from app.vector_store import VectorStore
    print("   ✅ Modules imported\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    exit(1)

print("2️⃣  Loading embedding model...")
try:
    embedder = create_embedder()
    dim = embedder.get_embedding_dimension()
    print(f"   ✅ Loaded (dim={dim})\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    exit(1)

print("3️⃣  Creating vector store...")
try:
    vector_store = VectorStore(embedding_dim=dim)
    print("   ✅ Created\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    exit(1)

print("4️⃣  Processing test document...")
try:
    import tempfile, os
    test_text = """Local AI Agent Project\n\nThis is a fully offline AI system.\nBuilt with LangChain, llama.cpp, and FAISS.\nPerfect for air-gapped environments."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(test_text)
        test_file = f.name
    doc = embedder.process_document(test_file)
    print(f"   ✅ Processed {len(doc['chunks'])} chunks\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    exit(1)

print("5️⃣  Generating embeddings...")
try:
    embedded_docs = embedder.embed_documents([doc])
    vector_store.add_documents(embedded_docs)
    print("   ✅ Generated & indexed\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    os.unlink(test_file)
    exit(1)

print("6️⃣  Testing search...")
try:
    queries = ["What is this?", "Is it offline?", "What tech?"]
    for query in queries:
        q_emb = embedder.embed_text(query)
        results = vector_store.search(q_emb, k=1)
        if results:
            print(f"   Query: '{query}'")
            print(f"   → {results[0]['text'][:50]}...")
            print(f"   → Score: {results[0]['score']:.3f}\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    os.unlink(test_file)
    exit(1)

os.unlink(test_file)
print("="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print("\n✅ Verified: Processing • Embeddings • Indexing • Search")
print("\n📌 Project is WORKING!\n")
