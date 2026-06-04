# Technical Summary

- The app uses a local RAG pipeline: `PyPDFLoader` reads the Upwork API PDF, LangChain splits it into 500-character chunks with 50-character overlap, FAISS stores vectors, and DeepInfra generates final answers from retrieved snippets only.
- Chunk overlap matters because technical answers often cross chunk boundaries. For example, an endpoint name, parameter list, and warning may be split apart; overlap preserves enough neighboring text so retrieval does not lose key context from code snippets or OAuth examples.
- A key difficulty is hallucination control. The system prompt explicitly requires the fallback sentence when the retrieved chunks do not contain the answer, and the UI displays exact snippets so every answer can be audited.
- Another difficulty is API latency. The app measures the DeepInfra request time separately and displays it, which makes slow hosted-model responses visible during evaluation.
- I used GPT/Codex to help structure the codebase, identify the required deliverables, and draft clear guardrails; I kept the implementation small enough that I can explain each function line by line.

## Why I am a strong fit for the ProAnalyst AI team

- I build AI systems with practical guardrails: source display, local vector storage, environment-based secrets, and explicit failure behavior.
- I focus on explainability and maintainability, which matters for client-facing technical support tools where every generated answer needs to be trusted.
- I can move from requirements to working product quickly while still checking edge cases such as missing evidence, latency, and documentation extraction quality.
