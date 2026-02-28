# AsyncReview Local Analysis Runbook

## Purpose

Learn how to integrate a new database backend using a pre-existing adapter pattern. This runbook uses sequential prompts in `cr ask` to build cumulative understanding of integration points, behaviors, boundaries, and tests.

## Setup

```bash
make analyze-mem0
```

## Prompts

### Phase 1: Map the Pattern

1. "What is the base class or interface for vector store backends? Show me the abstract methods and their signatures."

2. "Show me one complete concrete implementation of a vector store - pick the simplest one. Walk through how it implements each abstract method."

3. "How does the config system work for vector stores? Trace from a user passing `vector_store` config to a concrete backend being instantiated."

### Phase 2: Trace the Wiring

4. "How are vector store backends registered and resolved? Show me the factory, registry, or mapping that connects a config string to a concrete class."

5. "What is the full initialization chain? From `Memory()` constructor down to a vector store instance being ready to use."

6. "Are there any cross-cutting concerns - connection pooling, retry logic, lifecycle hooks, or cleanup methods that a new backend must handle?"

### Phase 3: Analyze Boundaries

7. "What data types cross the vector store boundary? Show me the models/dataclasses for documents, vectors, search results, and metadata."

8. "How does error handling work? Are there custom exceptions a backend is expected to raise, or does each implementation handle errors independently?"

9. "Are there any leaky abstractions - places where backend-specific behavior bleeds into the caller or where the base class makes assumptions about the underlying store?"

### Phase 4: Study the Tests

10. "How are vector store backends tested? Show me the test structure - are there shared test suites, fixtures, or contract tests that all backends must pass?"

11. "What mocking patterns are used? How do tests isolate the backend from its external dependency?"

### Phase 5: Build the Checklist

12. "If I were adding a new vector store backend called 'mytable', give me the exact checklist: every file I'd create, every file I'd modify, every registration step, every config class, and every test file. Be exhaustive."
