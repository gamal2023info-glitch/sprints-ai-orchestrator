# LangChain & LangGraph (PLACEHOLDER CONTENT - replace with real Sprints curriculum)

## RAG pipeline
Load documents, split into chunks (Sprints standard: 500 tokens with 50 tokens overlap),
embed the chunks, store them in a vector database such as ChromaDB, then retrieve the
top-k most similar chunks for each question.

## LangGraph state machines
A StateGraph is built from nodes (functions) and edges. State is a TypedDict that every
node reads and updates. Conditional edges route execution dynamically, enabling loops
such as self-reflection in Agentic RAG.

## Checkpointing and memory
A checkpointer such as MemorySaver stores state per thread_id, so each learner keeps an
isolated conversation across turns. Human-in-the-loop uses the interrupt primitive to
pause the graph until a person resumes it with a decision.
