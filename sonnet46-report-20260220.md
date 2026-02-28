# Runbook Report: mem0
Repository: /Users/dgordon/projects/mem0
Date: 2026-02-20 12:02
Questions answered: 15/15

## Q1: What is this project? What problem does it solve and how is the codebase organized at a high level?

## Mem0 ("mem-zero")

### What is this project?

Mem0 is an **intelligent memory layer for AI assistants and agents** that enables personalized, persistent, and context-aware AI interactions. It is available both as an open-source Python/JavaScript SDK (`pip install mem0ai`) and as a managed cloud platform. The project is backed by Y Combinator (S24) and has demonstrated +26% accuracy over OpenAI Memory on the LOCOMO benchmark, 91% faster responses than full-context approaches, and 90% lower token usage.

### What problem does it solve?

Current LLM-based AI systems are **stateless by default** — each conversation starts fresh with no memory of prior interactions, user preferences, or learned context. Mem0 solves this by providing:

1. **Persistent Multi-Level Memory**: Retains User, Session, and Agent state across conversations, enabling genuine personalization.
2. **Efficiency**: Instead of stuffing entire conversation histories into prompts (expensive and slow), Mem0 retrieves only the most relevant memories — 90% fewer tokens and 91% faster than full-context approaches.
3. **Adaptive Learning**: The system continuously learns and updates memories over time, adapting to individual user behavior.
4. **Developer Simplicity**: Provides a unified API that abstracts away the complexity of managing vector databases, graph databases, LLM calls, and embeddings.

**Use cases**: Customer support chatbots (recall past tickets), AI assistants (consistent context-rich conversations), healthcare (patient preferences/history), autonomous agents (persistent state management).

### How is the codebase organized?

The codebase is organized as the `mem0` Python package with the following top-level modules:

#### Core Logic
- **`mem0/memory/`** — The heart of the system
  - `main.py`: The primary `Memory` class — entry point for all operations (`add`, `search`, `update`, `delete`, `get_all`, `history`). Orchestrates LLMs, embeddings, vector stores, and graph backends. Also includes an `AsyncMemory` class for concurrent workloads.
  - `base.py`: Abstract `MemoryBase` class defining the standard memory interface.
  - `graph_memory.py`: Graph-based memory using Neo4j — extracts entities/relationships and builds knowledge graphs from memories.
  - `kuzu_memory.py`: Alternative embedded graph database (Kùzu) backend.
  - `memgraph_memory.py`: Memgraph-based graph memory backend.
  - `storage.py`: SQLite-based local storage for memory metadata/history.
  - `utils.py`, `telemetry.py`, `setup.py`: Utilities, analytics, and initialization helpers.

#### Integration Layers (pluggable backends)
- **`mem0/llms/`** — LLM integrations: OpenAI, Anthropic (Claude), Azure OpenAI, Google Gemini, Groq, Ollama, Deepseek, Together, LiteLLM, vLLM, xAI, LM Studio, Sarvam, and LangChain-compatible models.
- **`mem0/embeddings/`** — Embedding model integrations: OpenAI, Azure OpenAI, HuggingFace, Google Gemini, AWS Bedrock, Ollama, FastEmbed, VertexAI, Together, LM Studio, LangChain, and a mock embedder for testing.
- **`mem0/vector_stores/`** — Vector database backends (~20 integrations): Qdrant, Pinecone, Chroma, Weaviate, Milvus, MongoDB, PostgreSQL (pgvector), Redis, Elasticsearch, OpenSearch, Faiss, Azure AI Search, Azure MySQL, Supabase, Databricks, Cassandra, Baidu, S3 Vectors, Upstash, Valkey, Vertex AI, and LangChain-compatible stores.
- **`mem0/graphs/`** — Graph store backends: Neo4j (primary), AWS Neptune (via `mem0/graphs/neptune/`), with utilities for entity/relationship management.
- **`mem0/reranker/`** — Result reranking integrations: Cohere, HuggingFace cross-encoder, sentence-transformer, LLM-based reranker, and a zero-entropy reranker.

#### Configuration & Client
- **`mem0/configs/`** — Pydantic configuration classes for all components (LLMs, embeddings, vector stores, rerankers). Also contains `prompts.py` with system prompts used by the memory extraction pipeline.
- **`mem0/client/`** — Hosted platform client (`MemoryClient`) for interacting with the Mem0 managed cloud API, plus project management utilities.
- **`mem0/proxy/`** — Proxy functionality for routing requests.
- **`mem0/utils/`** — Factory classes for instantiating the correct backend based on configuration.

#### Tests
- **`tests/`** — Comprehensive test suite organized by component: `tests/llms/`, `tests/embeddings/`, `tests/vector_stores/`, `tests/memory/`, `tests/configs/`.

### Architecture summary

The `Memory` class serves as the central orchestrator. When you call `memory.add(messages, user_id=...)`:
1. An **LLM** extracts discrete memory facts from the conversation.
2. An **embedding model** converts those facts into vectors.
3. The vectors are stored/updated in a **vector store** for semantic similarity search.
4. Optionally, entities and relationships are also stored in a **graph store** for structured knowledge retrieval.
5. On `memory.search(query, user_id=...)`, relevant memories are retrieved by embedding similarity (and optionally reranked), then injected into the AI's context window.

This pluggable architecture means users can swap any backend (e.g., use Pinecone instead of Qdrant, Anthropic instead of OpenAI) via configuration without changing application code.

## Q2: How does this project handle pluggable backends or providers? What extension points exist for databases, vector stores, or storage?


## How Mem0 Handles Pluggable Backends and Providers

Mem0 uses a **factory + abstract base class pattern** to support pluggable backends across five major subsystems: vector stores, LLMs, embedders, graph stores, and rerankers. Every component is swappable by changing a provider name string in configuration.

---

### 1. Architecture Overview: Factory + ABC Pattern

The core mechanism lives in **`mem0/utils/factory.py`**, which provides five factory classes:

- **`LlmFactory`**
- **`EmbedderFactory`**
- **`VectorStoreFactory`**
- **`GraphStoreFactory`**
- **`RerankerFactory`**

Each factory holds a `provider_to_class` dictionary mapping a provider name string to a dotted module path and a config class. The `load_class()` helper uses `importlib.import_module()` to dynamically load the appropriate implementation class at runtime:

```python
def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

This means new providers can be added without changing the core memory logic — only the factory mapping and a new module need to be created.

---

### 2. Abstract Base Classes (Extension Points)

Each subsystem defines an abstract base class that all providers must implement:

**Vector Stores** (`mem0/vector_stores/base.py` → `VectorStoreBase`):
- Abstract methods: `create_col`, `insert`, `search`, `delete`, `update`, `get`, `list_cols`, `delete_col`, `col_info`, `list`, `reset`

**Embeddings** (`mem0/embeddings/base.py` → `EmbedderBase`):
- Abstract methods: `embed(text)`

**LLMs** (`mem0/llms/base.py` → `LLMBase`):
- Abstract methods: `generate_response(messages, ...)`

**Rerankers** (`mem0/reranker/base.py` → `RerankerBase`):
- Abstract methods: `rerank(query, documents, top_n)`

To add a custom backend, implement the corresponding ABC, place it in the appropriate module, and register it in the factory.

---

### 3. Vector Store Providers (24+ backends)

Configured via `VectorStoreConfig` (provider string + config dict). The `_provider_configs` mapping in `mem0/vector_stores/configs.py` registers all supported providers:

| Provider key | Implementation |
|---|---|
| `qdrant` | Default |
| `chroma` | ChromaDB |
| `pgvector` | PostgreSQL |
| `pinecone` | Pinecone |
| `mongodb` | MongoDB Atlas |
| `milvus` | Milvus |
| `cassandra` | Apache Cassandra |
| `redis` | Redis |
| `valkey` | Valkey |
| `elasticsearch` | Elasticsearch |
| `opensearch` | OpenSearch |
| `weaviate` | Weaviate |
| `faiss` | FAISS (local) |
| `supabase` | Supabase |
| `azure_ai_search` | Azure AI Search |
| `azure_mysql` | Azure MySQL |
| `databricks` | Databricks Vector Search |
| `vertex_ai_vector_search` | Google Vertex AI |
| `upstash_vector` | Upstash |
| `baidu` | Baidu VectorDB |
| `neptune` | AWS Neptune Analytics |
| `langchain` | Any LangChain vector store |
| `s3_vectors` | AWS S3 Vectors |

Each provider has its own config class (e.g., `QdrantConfig`, `PineconeConfig`) loaded dynamically from `mem0/configs/vector_stores/<provider>.py`. The `VectorStoreConfig.validate_and_create_config` model validator dynamically imports and instantiates the correct config class at validation time.

---

### 4. LLM Providers (17 providers)

Configured via `LlmConfig`. Supported providers registered in `LlmFactory.provider_to_class`:

`openai`, `azure_openai`, `openai_structured`, `azure_openai_structured`, `anthropic`, `ollama`, `groq`, `together`, `aws_bedrock`, `litellm`, `gemini`, `deepseek`, `xai`, `sarvam`, `lmstudio`, `vllm`, `langchain`

The `langchain` provider acts as a universal pass-through, allowing any LangChain-compatible LLM to be used.

---

### 5. Embedding Providers (11 providers)

Configured via `EmbedderConfig`. Supported providers validated in `EmbedderConfig`:

`openai`, `azure_openai`, `ollama`, `huggingface`, `gemini`, `vertexai`, `together`, `lmstudio`, `langchain`, `aws_bedrock`, `fastembed`

---

### 6. Graph Store Providers

Configured via `GraphStoreConfig` (in `mem0/graphs/configs.py`). Supports:
- **Neo4j** (`neo4j`) — via `Neo4jConfig` with url/username/password
- **Kuzu** — embedded graph database (used in `mem0/memory/kuzu_memory.py`)
- **Memgraph** — via `mem0/memory/memgraph_memory.py`
- **AWS Neptune** — via `mem0/graphs/neptune/` (both Neptune Database and Neptune Analytics)

---

### 7. Reranker Providers

Configured via `RerankerConfig`. Supported in `RerankerFactory`:

| Provider | Class |
|---|---|
| `cohere` | CohereReranker |
| `sentence_transformer` | SentenceTransformerReranker |
| `huggingface` | HuggingFaceReranker |
| `zero_entropy` | ZeroEntropyReranker |
| `llm` | LLMReranker (uses any configured LLM) |

---

### 8. Top-Level Configuration: `MemoryConfig`

Everything is wired together in `mem0/configs/base.py` via `MemoryConfig`:

```python
class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig   # pluggable vector store
    llm: LlmConfig                   # pluggable LLM
    embedder: EmbedderConfig         # pluggable embedder
    graph_store: GraphStoreConfig    # pluggable graph store
    reranker: Optional[RerankerConfig]  # optional pluggable reranker
    history_db_path: str             # SQLite for operation history
    version: str                     # API version
    custom_fact_extraction_prompt: Optional[str]  # custom prompts
    custom_update_memory_prompt: Optional[str]
```

Users configure backends by passing this config object (or a dict) to `Memory()`.

---

### 9. How to Add a New Provider

1. Create a new implementation class in the appropriate module (e.g., `mem0/vector_stores/myprovider.py`) extending `VectorStoreBase`
2. Create a config class in `mem0/configs/vector_stores/myprovider.py`
3. Register the provider in `VectorStoreConfig._provider_configs` (or the factory's `provider_to_class` dict for LLMs/embedders)
4. Use it via: `MemoryConfig(vector_store=VectorStoreConfig(provider="myprovider", config={...}))`

---

### 10. Special Integrations

- **LangChain pass-through**: Both the LLM and vector store layers support `langchain` as a provider, enabling integration with any LangChain-compatible component
- **LiteLLM**: The `litellm` LLM provider enables routing to 100+ models through a unified interface
- **Mock/Test providers**: `mem0/embeddings/mock.py` provides a `MockEmbeddings` class used in testing (also referenced in the factory for fallback)
- **SQLite history**: Operation history (add/update/delete audit trail) is always stored in a local SQLite database (`SQLiteManager`), separate from the pluggable vector store


## Q3: What interface or base abstraction do vector store backends implement? What are the method signatures and return types?

## Vector Store Base Abstraction

All vector store backends in Mem0 implement the abstract base class **`VectorStoreBase`**, defined in `mem0/vector_stores/base.py`. It uses Python's `abc.ABC` and `@abstractmethod` decorators to enforce the interface.

---

### Full Base Class Source (`mem0/vector_stores/base.py`)

    from abc import ABC, abstractmethod

    class VectorStoreBase(ABC):
        @abstractmethod
        def create_col(self, name, vector_size, distance):
            """Create a new collection."""
            pass

        @abstractmethod
        def insert(self, vectors, payloads=None, ids=None):
            """Insert vectors into a collection."""
            pass

        @abstractmethod
        def search(self, query, vectors, limit=5, filters=None):
            """Search for similar vectors."""
            pass

        @abstractmethod
        def delete(self, vector_id):
            """Delete a vector by ID."""
            pass

        @abstractmethod
        def update(self, vector_id, vector=None, payload=None):
            """Update a vector and its payload."""
            pass

        @abstractmethod
        def get(self, vector_id):
            """Retrieve a vector by ID."""
            pass

        @abstractmethod
        def list_cols(self):
            """List all collections."""
            pass

        @abstractmethod
        def delete_col(self):
            """Delete a collection."""
            pass

        @abstractmethod
        def col_info(self):
            """Get information about a collection."""
            pass

        @abstractmethod
        def list(self, filters=None, limit=None):
            """List all memories."""
            pass

        @abstractmethod
        def reset(self):
            """Reset by delete the collection and recreate it."""
            pass

---

### Method Signatures and Return Types

The base class does not include explicit return type annotations, but the concrete implementations (e.g., `Qdrant`) reveal the expected types:

| Method | Signature | Return Type | Description |
|--------|-----------|-------------|-------------|
| `create_col` | `create_col(self, name, vector_size, distance)` | `None` | Create a new collection |
| `insert` | `insert(self, vectors, payloads=None, ids=None)` | `None` | Insert vectors into a collection |
| `search` | `search(self, query, vectors, limit=5, filters=None)` | `list` | Search for similar vectors; returns a list of scored result points |
| `delete` | `delete(self, vector_id)` | `None` | Delete a vector by ID |
| `update` | `update(self, vector_id, vector=None, payload=None)` | `None` | Update a vector and its payload |
| `get` | `get(self, vector_id)` | `dict` or `None` | Retrieve a single vector by ID; returns `None` if not found |
| `list_cols` | `list_cols(self)` | `list` | List all collections |
| `delete_col` | `delete_col(self)` | `None` | Delete a collection |
| `col_info` | `col_info(self)` | `dict` | Get metadata/info about a collection |
| `list` | `list(self, filters=None, limit=None)` | `list` | List all memories/vectors, optionally filtered |
| `reset` | `reset(self)` | `None` | Reset by deleting and recreating the collection |

---

### Notes on Return Types (from Qdrant implementation)

- **`search`** — Returns `hits.points` (a list of scored points from the query result)
- **`get`** — Returns the first element of the retrieved results, or `None`
- **`list_cols`** — Returns a collections response object (iterable of collection names)
- **`col_info`** — Returns a collection info/metadata object
- **`list`** — Returns a scroll result (list of vectors/payloads)
- **`insert`, `delete`, `update`, `create_col`, `delete_col`, `reset`** — All return `None` (side-effect operations)

---

### Supported Backends

The following 23 providers implement `VectorStoreBase` (registered in `mem0/vector_stores/configs.py`):

`qdrant`, `chroma`, `pgvector`, `pinecone`, `mongodb`, `milvus`, `baidu`, `cassandra`, `neptune`, `upstash_vector`, `azure_ai_search`, `azure_mysql`, `redis`, `valkey`, `databricks`, `elasticsearch`, `vertex_ai_vector_search`, `opensearch`, `supabase`, `weaviate`, `faiss`, `langchain`, `s3_vectors`


## Q4: What data types, models, or structures are used to pass data in and out of vector store methods? How are results normalized across different backends?

## Data Types, Models, and Structures for Vector Store Methods

### Abstract Interface (`VectorStoreBase`)

The base class in `mem0/vector_stores/base.py` defines the contract with plain Python types:

- **`insert(vectors, payloads=None, ids=None)`** — `vectors` is a `list` of embedding floats, `payloads` is a `list` of dicts (metadata), `ids` is a `list` of string/int IDs.
- **`search(query, vectors, limit=5, filters=None)`** — `query` is a `str`, `vectors` is a `list` (the query embedding), `filters` is a `dict`.
- **`update(vector_id, vector=None, payload=None)`** — `vector_id` is an int/str, `vector` is a `list`, `payload` is a `dict`.
- **`get(vector_id)`**, **`delete(vector_id)`** — `vector_id` is an int/str.
- **`list(filters=None, limit=None)`** — `filters` is a `dict`.

---

### The `OutputData` Pydantic Model (Primary Normalization Structure)

The vast majority of backends define a **local `OutputData` Pydantic model** (in `chroma.py`, `pinecone.py`, `pgvector.py`, `milvus.py`, `mongodb.py`, `elasticsearch.py`, `weaviate.py`, `faiss.py`, `supabase.py`, and others):

```python
class OutputData(BaseModel):
    id: Optional[str]      # memory/record ID
    score: Optional[float] # similarity/distance score from search
    payload: Optional[Dict] # metadata dict (contains "data", "hash", "user_id", etc.)
```

This model is **defined independently in each backend module** (not shared from a central location), but is structurally identical across all backends that use it.

---

### Normalization via `_parse_output()`

Each backend implements a `_parse_output()` helper method that converts backend-native result objects into a list of `OutputData` instances. Examples:

- **Chroma**: Raw dict with `ids`, `distances`, `metadatas` lists → iterates and constructs `OutputData` per entry.
- **Pinecone**: `Vector` object or list of match dicts → single `OutputData` (for `get`) or list (for `search`), mapping `metadata` → `payload`.
- **MongoDB**: Pipeline result dicts → `OutputData(id=..., score=..., payload=...)`.
- **Milvus**, **Elasticsearch**, **Weaviate**, **FAISS**, **Supabase**: Similar pattern — each backend's raw hits/docs are mapped to `OutputData(id, score, payload)`.

---

### Exceptions / Deviations

- **Qdrant** (`mem0/vector_stores/qdrant.py`): Does **not** use `OutputData`. `search()` returns native Qdrant `PointStruct` objects (from `qdrant_client.models`), which carry `.id`, `.payload`, and `.score` attributes. `get()` returns a single `PointStruct` or `None`. `list()` returns a Qdrant scroll tuple `(points, next_offset)`.
- **Redis** (`mem0/vector_stores/redis.py`): Uses a plain Python class `MemoryResult` instead of a Pydantic model:
  ```python
  class MemoryResult:
      def __init__(self, id: str, payload: dict, score: float = None): ...
  ```
  It has the same duck-typed interface (`.id`, `.payload`, `.score`) as `OutputData`.

---

### How the Memory Layer Consumes Results

`mem0/memory/main.py` uses vector store results with attribute access, relying on duck typing:

```python
existing_memories = self.vector_store.search(...)
for mem in existing_memories:
    retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})
```

And when building the final `MemoryItem` output:
```python
MemoryItem(
    id=memory.id,
    memory=memory.payload.get("data", ""),
    hash=memory.payload.get("hash"),
    created_at=memory.payload.get("created_at"),
    updated_at=memory.payload.get("updated_at"),
)
```

This means all backends must expose `.id`, `.payload` (as a dict), and optionally `.score` on their result objects, regardless of whether they use `OutputData`, `PointStruct`, or `MemoryResult`.

---

### Summary Table

| Backend | Return Type from `search()` | Normalization Class |
|---|---|---|
| Chroma | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| Pinecone | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| PGVector | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| Milvus | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| MongoDB | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| Elasticsearch | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| Weaviate | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| FAISS | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| Supabase | `List[OutputData]` | `OutputData(BaseModel)` + `_parse_output()` |
| Qdrant | `List[PointStruct]` | Native Qdrant objects (no `OutputData`) |
| Redis | `List[MemoryResult]` | Plain Python class (duck-typed to same interface) |


## Q5: How does a user specify which vector store backend to use? What does the configuration look like?

## Specifying the Vector Store Backend in Mem0

### Overview

A user specifies which vector store backend to use via the `MemoryConfig` object, which is passed to `Memory()` at construction time. The relevant field is `vector_store`, which accepts a `VectorStoreConfig` instance (from `mem0.vector_stores.configs`).

---

### The Configuration Structure

```python
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig

config = MemoryConfig(
    vector_store=VectorStoreConfig(
        provider="<provider_name>",   # e.g. "qdrant", "chroma", "pinecone", etc.
        config={
            # provider-specific key/value pairs
        }
    )
)
m = Memory.from_config(config)
```

---

### `VectorStoreConfig` fields

Defined in `mem0/vector_stores/configs.py`:

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"qdrant"` | The backend to use (e.g. `"qdrant"`, `"chroma"`, `"pinecone"`) |
| `config` | `dict` or provider-specific config class | `None` | Provider-specific configuration options |

When the `VectorStoreConfig` is validated (via Pydantic `model_validator`), it dynamically imports the correct provider config class (e.g., `QdrantConfig`, `ChromaDbConfig`, `PineconeConfig`) and instantiates it with the values from the `config` dict.

---

### Supported Providers

The full list of supported `provider` values:

| Provider string | Config class |
|---|---|
| `qdrant` | `QdrantConfig` |
| `chroma` | `ChromaDbConfig` |
| `pgvector` | `PGVectorConfig` |
| `pinecone` | `PineconeConfig` |
| `mongodb` | `MongoDBConfig` |
| `milvus` | `MilvusDBConfig` |
| `baidu` | `BaiduDBConfig` |
| `cassandra` | `CassandraConfig` |
| `neptune` | `NeptuneAnalyticsConfig` |
| `upstash_vector` | `UpstashVectorConfig` |
| `azure_ai_search` | `AzureAISearchConfig` |
| `azure_mysql` | `AzureMySQLConfig` |
| `redis` | `RedisDBConfig` |
| `valkey` | `ValkeyConfig` |
| `databricks` | `DatabricksConfig` |
| `elasticsearch` | `ElasticsearchConfig` |
| `vertex_ai_vector_search` | `GoogleMatchingEngineConfig` |
| `opensearch` | `OpenSearchConfig` |
| `supabase` | `SupabaseConfig` |
| `weaviate` | `WeaviateConfig` |
| `faiss` | `FAISSConfig` |
| `langchain` | `LangchainConfig` |
| `s3_vectors` | `S3VectorsConfig` |

---

### Configuration Examples

#### Qdrant (default, local file-based)
```python
VectorStoreConfig(
    provider="qdrant",
    config={
        "collection_name": "mem0",
        "embedding_model_dims": 1536,
        "path": "/tmp/qdrant"        # local persistent storage
    }
)
```
Qdrant also supports `host`+`port` (self-hosted server) or `url`+`api_key` (Qdrant Cloud). Exactly one of `path`, `host`+`port`, or `url`+`api_key` must be provided.

#### Chroma (local)
```python
VectorStoreConfig(
    provider="chroma",
    config={
        "collection_name": "mem0",
        "path": "/tmp/chromadb"
    }
)
```
For a remote Chroma server use `host`+`port`; for Chroma Cloud use `api_key`+`tenant`.

#### Pinecone (cloud)
```python
VectorStoreConfig(
    provider="pinecone",
    config={
        "collection_name": "mem0",
        "embedding_model_dims": 1536,
        "api_key": "YOUR_PINECONE_API_KEY",
        "serverless_config": {"cloud": "aws", "region": "us-east-1"}
    }
)
```
`PINECONE_API_KEY` environment variable is also accepted.

#### PGVector (PostgreSQL)
```python
VectorStoreConfig(
    provider="pgvector",
    config={
        "collection_name": "mem0",
        "embedding_model_dims": 1536,
        "dbname": "postgres",
        "user": "postgres",
        "password": "secret",
        "host": "localhost",
        "port": 5432
    }
)
```
Alternatively supply a `connection_string` or a `connection_pool` object.

---

### Full Usage Example

```python
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig

config = MemoryConfig(
    vector_store=VectorStoreConfig(
        provider="qdrant",
        config={
            "collection_name": "my_memories",
            "embedding_model_dims": 1536,
            "host": "localhost",
            "port": 6333,
        }
    )
)

m = Memory.from_config(config)
```

---

### How It's Wired Internally

1. `MemoryConfig` (in `mem0/configs/base.py`) holds a `vector_store: VectorStoreConfig` field (defaulting to Qdrant).
2. `VectorStoreConfig` (in `mem0/vector_stores/configs.py`) validates the `provider` string and dynamically imports + instantiates the correct per-provider Pydantic config class from `mem0/configs/vector_stores/<provider>.py`.
3. `VectorStoreFactory` (in `mem0/utils/factory.py`) then uses `provider` to import the matching implementation class from `mem0/vector_stores/<provider>.py` and passes the resolved config to it.
4. Each provider config class is a **Pydantic model** with its own field validation (e.g., `QdrantConfig` requires one of `path`, `host`+`port`, or `url`+`api_key`).


## Q6: How does the system validate, resolve, and instantiate a vector store from configuration? Trace the full path from user config to a live backend instance.


## Full Path: User Config → Live Vector Store Backend Instance

The system follows a multi-layered pipeline: **config definition → Pydantic validation → provider-specific config resolution → factory instantiation → live backend connection**.

---

### Step 1: User Provides Configuration

The user creates (or passes a dict to) a `MemoryConfig` object, defined in `mem0/configs/base.py`:

```python
from mem0.configs.base import MemoryConfig
config = MemoryConfig(
    vector_store={
        "provider": "qdrant",
        "config": {"collection_name": "my_memories", "url": "http://localhost:6333", "api_key": "..."}
    }
)
# OR via the classmethod:
memory = Memory.from_config({
    "vector_store": {"provider": "qdrant", "config": {"url": "...", "api_key": "..."}}
})
```

`MemoryConfig` is a Pydantic `BaseModel` with a `vector_store` field typed as `VectorStoreConfig`:

```python
class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,  # defaults to Qdrant
    )
    ...
```

---

### Step 2: `VectorStoreConfig` — The Central Validation & Resolution Hub

`VectorStoreConfig` lives in `mem0/vector_stores/configs.py` and is a Pydantic model with two fields:

```python
class VectorStoreConfig(BaseModel):
    provider: str = Field(default="qdrant")  # e.g., "qdrant", "chroma", "pinecone"
    config: Optional[Dict] = Field(default=None)  # raw dict of provider-specific settings
```

It maintains a mapping of every supported provider to its specific config class name:

```python
_provider_configs: Dict[str, str] = {
    "qdrant": "QdrantConfig",
    "chroma": "ChromaDbConfig",
    "pgvector": "PGVectorConfig",
    "pinecone": "PineconeConfig",
    "mongodb": "MongoDBConfig",
    "milvus": "MilvusDBConfig",
    "redis": "RedisDBConfig",
    "elasticsearch": "ElasticsearchConfig",
    "weaviate": "WeaviateConfig",
    "faiss": "FAISSConfig",
    # ... 20+ more providers
}
```

#### The Key: `@model_validator(mode="after")` — `validate_and_create_config`

This Pydantic validator runs **automatically after model initialization** and performs all validation, resolution, and config instantiation:

```python
@model_validator(mode="after")
def validate_and_create_config(self) -> "VectorStoreConfig":
    provider = self.provider
    config = self.config

    # 1. VALIDATE: Check provider is supported
    if provider not in self._provider_configs:
        raise ValueError(f"Unsupported vector store provider: {provider}")

    # 2. RESOLVE: Dynamically import the provider-specific config class
    module = __import__(
        f"mem0.configs.vector_stores.{provider}",   # e.g., mem0.configs.vector_stores.qdrant
        fromlist=[self._provider_configs[provider]], # e.g., ["QdrantConfig"]
    )
    config_class = getattr(module, self._provider_configs[provider])  # e.g., QdrantConfig

    # 3. NORMALIZE: Convert None to empty dict
    if config is None:
        config = {}

    # 4. TYPE-CHECK: If config is already a typed config object, accept as-is
    if not isinstance(config, dict):
        if not isinstance(config, config_class):
            raise ValueError(f"Invalid config type for provider {provider}")
        return self

    # 5. DEFAULT PATH: Set a default file path for file-based stores if missing
    if "path" not in config and "path" in config_class.__annotations__:
        config["path"] = f"/tmp/{provider}"  # e.g., "/tmp/qdrant"

    # 6. INSTANTIATE: Replace raw dict with typed, validated config object
    self.config = config_class(**config)
    return self
```

---

### Step 3: Provider-Specific Config Validation (e.g., `QdrantConfig`)

Each provider's config class (in `mem0/configs/vector_stores/{provider}.py`) performs its own deep validation. For Qdrant (`mem0/configs/vector_stores/qdrant.py`):

```python
class QdrantConfig(BaseModel):
    collection_name: str = Field("mem0", ...)
    embedding_model_dims: Optional[int] = Field(1536, ...)
    client: Optional[QdrantClient] = Field(None, ...)
    host: Optional[str] = Field(None, ...)
    port: Optional[int] = Field(None, ...)
    path: Optional[str] = Field("/tmp/qdrant", ...)
    url: Optional[str] = Field(None, ...)
    api_key: Optional[str] = Field(None, ...)
    on_disk: Optional[bool] = Field(False, ...)

    @model_validator(mode="before")
    @classmethod
    def check_host_port_or_path(cls, values):
        # VALIDATES: one of (host+port), (url+api_key), or (path) must be present
        if not path and not (host and port) and not (url and api_key):
            raise ValueError("Either 'host' and 'port' or 'url' and 'api_key' or 'path' must be provided.")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values):
        # VALIDATES: no unknown fields passed in
        allowed_fields = set(cls.model_fields.keys())
        extra_fields = set(values.keys()) - allowed_fields
        if extra_fields:
            raise ValueError(f"Extra fields not allowed: {extra_fields}")
        return values
```

After this step, `VectorStoreConfig.config` holds a fully-validated, typed `QdrantConfig` (or equivalent) object.

---

### Step 4: `Memory.__init__` Calls the Factory

In `mem0/memory/main.py`, the `Memory` class receives the fully-validated `MemoryConfig` and delegates instantiation to `VectorStoreFactory`:

```python
class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        # Embedder first (needs vector store config for special cases like Upstash)
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,  # passed for Upstash special handling
        )

        # THE KEY CALL: vector store instantiation
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider,   # e.g., "qdrant"
            self.config.vector_store.config      # e.g., QdrantConfig(...)
        )

        self.collection_name = self.config.vector_store.config.collection_name
        ...
```

---

### Step 5: `VectorStoreFactory.create` — Dynamic Class Loading & Instantiation

`VectorStoreFactory` in `mem0/utils/factory.py` maintains a registry mapping provider names to their implementation class paths:

```python
class VectorStoreFactory:
    provider_to_class = {
        "qdrant":                  "mem0.vector_stores.qdrant.Qdrant",
        "chroma":                  "mem0.vector_stores.chroma.ChromaDB",
        "pgvector":                "mem0.vector_stores.pgvector.PGVector",
        "pinecone":                "mem0.vector_stores.pinecone.PineconeDB",
        "mongodb":                 "mem0.vector_stores.mongodb.MongoDB",
        "redis":                   "mem0.vector_stores.redis.RedisDB",
        "elasticsearch":           "mem0.vector_stores.elasticsearch.ElasticsearchDB",
        "weaviate":                "mem0.vector_stores.weaviate.Weaviate",
        "faiss":                   "mem0.vector_stores.faiss.FAISS",
        "milvus":                  "mem0.vector_stores.milvus.MilvusDB",
        "opensearch":              "mem0.vector_stores.opensearch.OpenSearchDB",
        # ... 20+ providers
    }

    @classmethod
    def create(cls, provider_name, config):
        # 1. RESOLVE: Look up the full dotted class path
        class_type = cls.provider_to_class.get(provider_name)
        if not class_type:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

        # 2. SERIALIZE: Convert typed Pydantic config to plain dict
        if not isinstance(config, dict):
            config = config.model_dump()  # e.g., {"collection_name": "mem0", "url": "...", ...}

        # 3. LOAD: Dynamically import the implementation class
        vector_store_instance = load_class(class_type)
        # load_class does: importlib.import_module("mem0.vector_stores.qdrant") → getattr(..., "Qdrant")

        # 4. INSTANTIATE: Call the class constructor with unpacked config dict
        return vector_store_instance(**config)  # → Qdrant(collection_name="mem0", url="...", ...)
```

The helper `load_class` uses Python's `importlib` for lazy, dynamic loading:

```python
def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)  # imports mem0.vector_stores.qdrant
    return getattr(module, class_name)             # returns the Qdrant class
```

---

### Step 6: Backend Class `__init__` — Live Connection Established

The implementation class (e.g., `Qdrant` in `mem0/vector_stores/qdrant.py`) receives the unpacked config fields as keyword arguments and creates the real backend connection:

```python
class Qdrant(VectorStoreBase):
    def __init__(self, collection_name, embedding_model_dims, client=None,
                 host=None, port=None, path=None, url=None, api_key=None, on_disk=False):
        if client:
            # Use a pre-existing client object
            self.client = client
        else:
            params = {}
            if api_key: params["api_key"] = api_key
            if url:     params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port

            if not params:
                # Local file-based mode
                params["path"] = path
                if not on_disk and os.path.exists(path):
                    shutil.rmtree(path)  # clear old data

            # LIVE CONNECTION: instantiate the real QdrantClient
            self.client = QdrantClient(**params)

        self.collection_name = collection_name
        self.create_col(embedding_model_dims, on_disk)  # create/verify collection
```

---

### Complete Flow Summary

```
User dict / MemoryConfig(vector_store={"provider": "qdrant", "config": {...}})
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  VectorStoreConfig (mem0/vector_stores/configs.py)              │
│  @model_validator("after"): validate_and_create_config          │
│  1. Check provider in _provider_configs dict                     │
│  2. __import__("mem0.configs.vector_stores.qdrant", ["QdrantConfig"]) │
│  3. Set default path if missing                                  │
│  4. self.config = QdrantConfig(**raw_dict)                       │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  QdrantConfig (mem0/configs/vector_stores/qdrant.py)            │
│  @model_validator("before"): check_host_port_or_path            │
│    → validates connection method is specified                    │
│  @model_validator("before"): validate_extra_fields              │
│    → rejects any unknown fields                                  │
│  Result: typed, validated QdrantConfig object                    │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Memory.__init__ (mem0/memory/main.py)                          │
│  VectorStoreFactory.create(                                      │
│      provider="qdrant",                                          │
│      config=<QdrantConfig object>                                │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  VectorStoreFactory (mem0/utils/factory.py)                      │
│  1. Lookup: "qdrant" → "mem0.vector_stores.qdrant.Qdrant"       │
│  2. config.model_dump() → plain Python dict                      │
│  3. load_class() → importlib.import_module + getattr            │
│  4. Qdrant(**config_dict)                                        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Qdrant.__init__ (mem0/vector_stores/qdrant.py)                  │
│  → QdrantClient(url=..., api_key=...) [or host/port or path]    │
│  → create_col(embedding_model_dims, on_disk)                     │
│  Result: LIVE, connected vector store backend instance           │
└─────────────────────────────────────────────────────────────────┘
```

**Key design patterns used:**
- **Pydantic model validators** (`@model_validator`) drive validation and config type coercion automatically at model construction time
- **Two-tier config system**: a generic `VectorStoreConfig` wrapper delegates to provider-specific configs (e.g., `QdrantConfig`)
- **Dynamic imports** (`__import__` and `importlib.import_module`) enable lazy loading — backends are only imported when needed
- **Factory pattern** (`VectorStoreFactory`) centralizes the provider-to-class registry and decouples config from instantiation
- **`model_dump()`** bridges Pydantic models to plain kwargs for the backend constructor call


## Q7: Walk through one concrete vector store implementation end-to-end. How does it handle initialization, connections, queries, inserts, deletes, and cleanup?

## Qdrant Vector Store — End-to-End Walkthrough

The file `mem0/vector_stores/qdrant.py` implements the `Qdrant` class, which extends `VectorStoreBase` (the abstract interface in `mem0/vector_stores/base.py`). Here is a complete, phase-by-phase walkthrough.

---

### 0. The Abstract Interface (`VectorStoreBase`)

Every vector store in Mem0 must implement:
`create_col`, `insert`, `search`, `delete`, `update`, `get`, `list_cols`, `delete_col`, `col_info`, `list`, `reset`.

---

### 1. Configuration (`QdrantConfig`)

File: `mem0/configs/vector_stores/qdrant.py`

The Pydantic config model captures:
- `collection_name` (default `"mem0"`)
- `embedding_model_dims` (default `1536`)
- `client` — an optional pre-existing `QdrantClient`
- `host` / `port` — for a self-hosted server
- `url` / `api_key` — for Qdrant Cloud
- `path` (default `"/tmp/qdrant"`) — for a local on-disk store
- `on_disk` (default `False`) — controls persistence

A `@model_validator` enforces that at least one of `(host+port)`, `(url+api_key)`, or `path` is provided.

---

### 2. Initialization (`__init__`)

```python
def __init__(self, collection_name, embedding_model_dims,
             client=None, host=None, port=None,
             path=None, url=None, api_key=None, on_disk=False):
```

**Three connection modes:**

1. **Pre-existing client** — if a `QdrantClient` object is passed directly, it is used as-is (`self.is_local = False`).

2. **Remote server** — if `host`+`port` or `url`+`api_key` are provided, they are assembled into a `params` dict and `QdrantClient(**params)` is called. `self.is_local = False`.

3. **Local path** — if none of the above are set, falls back to `path`. Sets `self.is_local = True`. If `on_disk=False` (in-memory/ephemeral mode) **and** the path directory already exists, it is deleted with `shutil.rmtree(path)` to ensure a clean slate.

After creating the client, the constructor calls `self.create_col(embedding_model_dims, on_disk)`.

---

### 3. Collection Creation (`create_col` + `_create_filter_indexes`)

```python
def create_col(self, vector_size, on_disk, distance=Distance.COSINE):
    response = self.list_cols()
    for collection in response.collections:
        if collection.name == self.collection_name:
            self._create_filter_indexes()
            return  # already exists — skip

    self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
    )
    self._create_filter_indexes()
```

- **Idempotent**: checks if the collection already exists before creating it.
- Creates a collection with cosine-distance vectors of the configured dimension.
- After creation (or if already existing), calls `_create_filter_indexes()`.

```python
def _create_filter_indexes(self):
    if self.is_local:
        return  # local Qdrant does not support payload indexes
    common_fields = ["user_id", "agent_id", "run_id", "actor_id"]
    for field in common_fields:
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field,
            field_schema="keyword"
        )
```

For remote instances, creates keyword indexes on the four standard Mem0 metadata fields so filtered queries are fast. Exceptions are silently swallowed (index may already exist).

---

### 4. Inserting Vectors (`insert`)

```python
def insert(self, vectors, payloads=None, ids=None):
    points = [
        PointStruct(
            id=idx if ids is None else ids[idx],
            vector=vector,
            payload=payloads[idx] if payloads else {},
        )
        for idx, vector in enumerate(vectors)
    ]
    self.client.upsert(collection_name=self.collection_name, points=points)
```

- Builds a list of `PointStruct` objects, each with an ID, a float-vector, and an optional metadata payload (dict).
- Uses `upsert` (insert-or-replace), so re-inserting the same ID is safe.
- If no `ids` list is given, uses the loop index as the ID.

---

### 5. Querying / Searching (`search` + `_create_filter`)

```python
def search(self, query, vectors, limit=5, filters=None):
    query_filter = self._create_filter(filters) if filters else None
    hits = self.client.query_points(
        collection_name=self.collection_name,
        query=vectors,          # the embedding vector
        query_filter=query_filter,
        limit=limit,
    )
    return hits.points
```

**Filter construction** (`_create_filter`):
```python
def _create_filter(self, filters):
    conditions = []
    for key, value in filters.items():
        if isinstance(value, dict) and "gte" in value and "lte" in value:
            conditions.append(FieldCondition(key=key, range=Range(gte=..., lte=...)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions)
```

- Supports both **exact-match** filters (`user_id="alice"`) and **range** filters (`{"gte": 0.5, "lte": 1.0}`).
- All conditions are ANDed via `Filter(must=conditions)`.

---

### 6. Deleting a Vector (`delete`)

```python
def delete(self, vector_id):
    self.client.delete(
        collection_name=self.collection_name,
        points_selector=PointIdsList(points=[vector_id]),
    )
```

Wraps the ID in a `PointIdsList` and calls the Qdrant delete API.

---

### 7. Updating a Vector (`update`)

```python
def update(self, vector_id, vector=None, payload=None):
    point = PointStruct(id=vector_id, vector=vector, payload=payload)
    self.client.upsert(collection_name=self.collection_name, points=[point])
```

Uses `upsert` again — the existing point with `vector_id` is replaced wholesale with the new vector and/or payload.

---

### 8. Retrieving a Single Vector (`get`)

```python
def get(self, vector_id):
    result = self.client.retrieve(
        collection_name=self.collection_name,
        ids=[vector_id],
        with_payload=True,
    )
    return result[0] if result else None
```

Fetches by exact ID and returns the point (including payload) or `None` if not found.

---

### 9. Listing Vectors (`list`)

```python
def list(self, filters=None, limit=100):
    query_filter = self._create_filter(filters) if filters else None
    result = self.client.scroll(
        collection_name=self.collection_name,
        scroll_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,   # no need to return raw vectors
    )
    return result
```

Uses Qdrant's scroll API for paginated listing. Returns payloads without the raw vectors.

---

### 10. Collection Introspection

```python
def list_cols(self):
    return self.client.get_collections()

def col_info(self):
    return self.client.get_collection(collection_name=self.collection_name)
```

---

### 11. Cleanup: Deleting the Collection (`delete_col`)

```python
def delete_col(self):
    self.client.delete_collection(collection_name=self.collection_name)
```

Drops the entire collection from the store.

---

### 12. Full Reset (`reset`)

```python
def reset(self):
    logger.warning(f"Resetting index {self.collection_name}...")
    self.delete_col()
    self.create_col(self.embedding_model_dims, self.on_disk)
```

Drops and immediately re-creates the collection (and its indexes), giving a clean empty store with the same configuration.

---

### Summary Data Flow

```
User calls Memory.add(text, user_id="alice")
  → LLM extracts facts → embedder produces vector
  → Qdrant.insert([vector], [{"user_id":"alice", "data":"..."}], [uuid])
     → PointStruct(id=uuid, vector=[...], payload={...})
     → client.upsert(...)

User calls Memory.search(query, user_id="alice")
  → embedder produces query vector
  → Qdrant.search(query, [vector], limit=5, filters={"user_id":"alice"})
     → _create_filter → Filter(must=[FieldCondition(user_id, MatchValue("alice"))])
     → client.query_points(...) → returns ranked Points

User calls Memory.delete(memory_id)
  → Qdrant.delete(memory_id)
     → client.delete(PointIdsList([memory_id]))
```


## Q8: What provider-specific concerns does this implementation handle that the base abstraction doesn't account for?

## Provider-Specific Concerns Beyond the Base Abstractions

The base abstractions in Mem0 are intentionally thin — they define only method signatures with no authentication, connection management, data format, or API-quirk handling. Individual implementations must handle a wide range of provider-specific concerns:

---

### 1. Authentication & Credential Management

The base classes have no auth mechanism. Every implementation manages its own credentials:

- **OpenAI** (`mem0/llms/openai.py`): Resolves credentials across `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_BASE_URL`, and `OPENROUTER_API_BASE` environment variables with priority ordering, and supports HTTP proxy configuration.
- **Anthropic** (`mem0/llms/anthropic.py`): Reads `ANTHROPIC_API_KEY` from env or config, initializes `anthropic.Anthropic` client directly.
- **AWS Bedrock** (`mem0/llms/aws_bedrock.py`): Handles full AWS IAM credential resolution (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`), catches `NoCredentialsError` and `UnauthorizedOperation`, and validates region-based model availability by listing foundation models.
- **Azure OpenAI** (`mem0/llms/azure_openai.py`): Handles Azure-specific endpoint URLs, deployment names, and API version parameters — entirely absent from the base.

---

### 2. Client Initialization & Connection Management

`LLMBase` and `VectorStoreBase` define no persistent client objects. Implementations each create their own SDK clients:

- **AWS Bedrock**: Uses two separate boto3 clients (`bedrock-runtime` for inference, `bedrock` for model listing), with a `_test_connection()` method to probe availability.
- **OpenAI**: Creates an `openai.OpenAI` client with optional proxy config from `http_client_proxies`.
- **Anthropic**: Maintains an `anthropic.Anthropic` client instance across calls.
- **Qdrant** (`mem0/vector_stores/qdrant.py`): Connects to either an in-memory or remote Qdrant instance depending on config, using `QdrantClient`.

---

### 3. Message Format Translation

`LLMBase.generate_response()` accepts messages as a generic list, but providers require different formats:

- **Anthropic**: Separates `"system"` role messages from the messages list and passes them as a separate `system` parameter to the API — the base class has no concept of this role distinction.
- **AWS Bedrock**: Dynamically selects a `_format_messages()` method based on the detected underlying provider (Anthropic, Cohere, Meta/Llama, Mistral, Amazon Titan, etc.) since each has its own API contract under the Bedrock umbrella.
- **OpenRouter (via OpenAI)**: Passes a `models` array and `route` selection parameter instead of a single `model` string, requiring the `model` field to be dropped from the standard request.

---

### 4. Model-Specific Behavior (Reasoning Models)

`LLMBase` includes a `_is_reasoning_model()` helper (checking for `o1`, `o3-mini`, `o3`, `gpt-5` series), but individual implementations must act on this:

- **OpenAI**: Skips `temperature`, `top_p`, `max_tokens`, and `stream` parameters for reasoning models, which don't support them.
- **OpenAI Structured** (`mem0/llms/openai_structured.py`): Handles structured JSON output via `response_format={"type": "json_schema", ...}` — a feature not in the base interface at all.

---

### 5. Tool/Function Calling Format

The base `generate_response()` signature doesn't define a tool-calling interface. Implementations add this themselves:

- **OpenAI**: Serializes tool definitions into OpenAI's `tools` array format and deserializes `tool_calls` responses by parsing `function.name` and `function.arguments` (using `extract_json()` for potentially malformed JSON strings).
- **Anthropic**: Passes `tools` and `tool_choice` parameters in Anthropic's format, and parses tool-use blocks from the response content differently than text blocks.
- **AWS Bedrock**: Maps tool support capability per underlying provider family via a `supports_tools` boolean flag.

---

### 6. Multi-Provider Routing (AWS Bedrock)

AWS Bedrock is unique in that it hosts multiple providers' models. The Bedrock implementation adds an entire layer the base doesn't anticipate:

- **`extract_provider()` regex**: Matches model IDs against 16+ known providers (Anthropic, Cohere, Meta, Mistral, Amazon, AI21, etc.)
- **Capability flags**: Sets `supports_tools`, `supports_vision`, `supports_streaming` per provider family
- **Dynamic message formatter assignment**: Points `_format_messages` at a provider-specific method at initialization time

---

### 7. Vector Store Index Configuration

`VectorStoreBase.create_col(name, vector_size, distance)` has three parameters. Implementations add far more:

- **Qdrant**: Configures HNSW indexing parameters (`m`, `ef_construct`), quantization settings, collection creation options, and Qdrant-specific distance metrics.
- **Pinecone** (`mem0/vector_stores/pinecone.py`): Handles serverless vs. pod-based index architecture, cloud/region selection, and Pinecone's eventual-consistency index creation (polling until ready).
- **Elasticsearch** (`mem0/vector_stores/elasticsearch.py`): Creates index mappings with `dense_vector` field type, configures similarity algorithms (`cosine`, `dot_product`, `l2_norm`), and handles index settings like `number_of_shards`.
- **Azure AI Search** (`mem0/vector_stores/azure_ai_search.py`): Defines HNSW algorithm configuration, vector search profiles, semantic configurations, and Azure-specific index schemas.
- **Redis** (`mem0/vector_stores/redis.py`): Creates a RediSearch index with `VECTOR` field type, configuring algorithm (`FLAT` vs `HNSW`) and distance metric as part of the index schema.

---

### 8. Result Normalization (Per-Backend)

The base abstraction defines no result format. Implementations each define their own `_parse_output()` method to normalize native results:

- **Most backends** (Chroma, Pinecone, MongoDB, Milvus, Elasticsearch, Weaviate, FAISS, Supabase): Define a local `OutputData(id, score, payload)` Pydantic model and implement `_parse_output()` to map provider-native result objects into it.
- **Qdrant**: Returns native `PointStruct` objects rather than `OutputData` — relying on duck typing (`.id`, `.payload`, `.score` attributes) to satisfy the `main.py` consumer.
- **Redis**: Defines a plain Python `MemoryResult` class (not Pydantic) with the same duck-typed interface.

---

### 9. Provider-Specific API Quirks

- **OpenAI/OpenRouter dual-mode**: The OpenAI implementation detects OpenRouter usage via env vars and switches request parameters (using `models[]` array, custom HTTP headers `HTTP-Referer` and `X-Title`, different `store` parameter handling).
- **Pinecone**: Must poll for index readiness after creation; the `create_col` method includes retry logic absent from the base.
- **Gemini** (`mem0/llms/gemini.py`): Uses a completely different SDK (`google.generativeai`) with its own content-part format for messages, requiring substantial transformation of the standard messages list.
- **Ollama** (`mem0/llms/ollama.py`): Targets a local HTTP endpoint (default `http://localhost:11434`), requires no API key, and may support a streaming-only response mode.
- **Sarvam** (`mem0/llms/sarvam.py`): Targets a custom base URL for an Indian-language-focused LLM provider, wrapping an OpenAI-compatible API with Sarvam-specific defaults.

---

### 10. Parameter Filtering & Validation

`LLMBase._validate_config()` is a near-empty placeholder. Implementations add real validation:

- **Anthropic**: Uses `_get_supported_params()` to filter kwargs to Anthropic-supported parameters, preventing invalid params from reaching the API.
- **OpenAI**: Conditionally omits `model` when using OpenRouter's `models` array; omits reasoning-model-incompatible params.
- **AWS Bedrock**: Validates model availability in the configured region before attempting inference.

---

### Summary

| Concern | Base Abstraction | Implementation Reality |
|---|---|---|
| Authentication | None | API key, AWS IAM, Azure AD, env var fallbacks |
| Client initialization | None | SDK-specific client objects with configs |
| Message format | Generic list | Anthropic: system separation; Bedrock: per-provider formatters |
| Tool calling | Not defined | Provider-specific tool formats + response parsing |
| Index configuration | 3 params | HNSW settings, shards, schemas, cloud regions |
| Result normalization | None | `OutputData`, `MemoryResult`, or native objects |
| Model behavior | `_is_reasoning_model()` helper | Param filtering, structured output, streaming modes |
| Multi-provider routing | None | Bedrock: detect Anthropic/Meta/Cohere/etc. under one API |
| Connection validation | None | AWS: connection test + model availability check |
| API quirks | None | OpenRouter headers, Pinecone polling, Gemini SDK format |


## Q9: How do vector store backends handle errors? Are there shared exception types, or does each backend handle failures independently?

## Vector Store Backend Error Handling in Mem0

### No Shared Exception Types

There are **no shared exception types** across vector store backends. The `VectorStoreBase` abstract base class (`mem0/vector_stores/base.py`) defines only abstract method signatures with no error handling whatsoever, and there is no shared exceptions module in the codebase. Each backend handles failures entirely independently.

---

### Common Patterns Across Backends

**1. Import Guard (Universal)**

Nearly every backend wraps its third-party library import in `try/except ImportError` and raises a descriptive `ImportError` with pip install instructions:

```python
try:
    from qdrant_client import QdrantClient
    ...
except ImportError:
    raise ImportError("The 'qdrant-client' library is required. Please install it using 'pip install qdrant-client'.")
```
This is the one consistent pattern across all backends.

**2. Logging + Re-raise (Most Backends)**

The dominant error-handling idiom is: catch a generic or specific exception, log it at `error` level, then re-raise:

```python
try:
    # operation
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

Cassandra, Databricks, OpenSearch, Supabase, Azure MySQL, Azure AI Search, and Valkey all follow this pattern extensively.

**3. Log-and-Swallow (Some Backends)**

A few backends log errors but do **not** re-raise, effectively swallowing failures:

- **Qdrant**: Index creation errors are silently swallowed with `logger.debug(f"Index for {field} might already exist: {e}")` — treated as a non-fatal best-effort operation.
- **Pinecone**: Some operations (e.g., `get`, `list`) catch `Exception` and log with `logger.error` but return `None`/empty without raising.
- **Baidu** reset: uses `logger.warning` and swallows reset errors.
- **Elasticsearch** search parsing: logs `warning` and continues processing.

**4. ValueError for State/Validation Issues (FAISS)**

FAISS uniquely raises `ValueError` for internal state invariants:
```python
raise ValueError("Collection not initialized. Call create_col first.")
raise ValueError("Vectors, payloads, and IDs must have the same length")
raise ValueError(f"Vector {vector_id} not found")
```

---

### Backend-Specific Exception Types

Several backends import and use their own SDK-specific exceptions:

| Backend | Exception Type | Source |
|---|---|---|
| **MongoDB** | `PyMongoError` | `pymongo.errors` |
| **Azure AI Search** | `ResourceNotFoundError` | `azure.core.exceptions` |
| **Valkey** | `ResponseError` (caught, converted to `ValueError`) | `valkey.exceptions` |
| **S3 Vectors** | `ClientError` (inspects error code) | `botocore.exceptions` |
| **Baidu** | `ServerError` | `pymochow.exception` |
| **Cassandra** | `json.JSONDecodeError` (nested within generic `Exception` handler) | `json` stdlib |
| **OpenSearch** | Raises `TimeoutError` for index creation timeout | stdlib |
| **Elasticsearch** | Catches `KeyError`, `TypeError`, then `Exception` sequentially | stdlib |
| **Databricks** | Raises `KeyError` for missing vector IDs | stdlib |

---

### Per-Backend Summary

| Backend | Primary Strategy | Re-raises? | Notes |
|---|---|---|---|
| **Qdrant** | `try/except Exception` with `logger.debug` | No (for index ops) | Most operation errors bubble up from client |
| **Chroma** | Minimal — import guard only | N/A | Almost no explicit error handling |
| **Pinecone** | `try/except Exception` + `logger.error` | No (some ops) | Swallows some failures, returns empty |
| **MongoDB** | `PyMongoError` + `logger.error` | Yes | Uses backend-specific type consistently |
| **Cassandra** | Generic `Exception` + `logger.error` | Yes (always) | Most thorough re-raising pattern |
| **Elasticsearch** | Sequential specific → generic catches | No (logs warning) | `KeyError` → `TypeError` → `Exception` |
| **OpenSearch** | Generic `Exception` + `logger.error` | Yes | Raises `TimeoutError` for index creation |
| **PGVector** | Context-manager based, `logger.error` | Yes | Uses psycopg context manager for rollback |
| **Azure AI Search** | `ResourceNotFoundError` + generic | Yes | Raises bare `Exception` for failed ops |
| **Azure MySQL** | Generic `Exception` + `logger.error` | Yes | Connection pool + context-manager rollback |
| **FAISS** | `ValueError` + `logger.warning` | Partial | `ValueError` for invariants, warning for I/O |
| **Milvus** | Minimal — import guard only | N/A | Relies on SDK raising its own errors |
| **Weaviate** | Minimal — import guard only | N/A | Relies on SDK raising its own errors |
| **Redis** | Minimal | N/A | Very little explicit error handling |
| **Valkey** | `ResponseError` → `ValueError` + `logger.exception` | Yes | Converts SDK error type to stdlib ValueError |
| **S3 Vectors** | `ClientError` with error code inspection | Partial | Handles NotFoundException specially |
| **Baidu** | `ServerError` + generic `Exception` | Yes (mostly) | Backend-specific type for table ops |
| **Databricks** | Generic `Exception` + `logger.error` | Yes | Very thorough, logs+raises everywhere |
| **Supabase** | Generic `Exception` + `logger.error` | Yes | Simple but consistent |
| **Neptune** | Import guard + `logger.error` (status checks) | No | Logs abnormal status without raising |
| **Langchain** | Generic `Exception` + `logger.error` | No (some ops) | Thin wrapper around Chroma |
| **Upstash** | `ValueError` for missing config/data | N/A | Raises ValueError for config validation |

---

### Summary

Each vector store backend handles errors **entirely independently** with no shared exception infrastructure. The only consistent cross-backend patterns are: (1) the import guard that raises `ImportError` with install instructions, and (2) the broad use of Python's standard `logging` module. Beyond that, error handling varies significantly: some backends (Cassandra, Databricks) are meticulous about logging and re-raising; others (Chroma, Milvus, Weaviate, Redis) add almost no error handling beyond what the underlying SDK provides; and a few (Pinecone, Qdrant index ops) silently swallow certain failures. There is no shared `MemoryError`, `VectorStoreError`, or any other project-defined exception class.


## Q10: What lifecycle concerns exist - connection management, resource cleanup, reconnection, or graceful shutdown?

## Lifecycle Concerns in Mem0

### 1. Connection Management

#### PostgreSQL / pgvector (`mem0/vector_stores/pgvector.py`)
- Uses **connection pooling** via `psycopg_pool.ConnectionPool` (psycopg3) or `psycopg2.pool.ThreadedConnectionPool` (psycopg2), with auto-detection and fallback between the two.
- Pool size is configurable via `minconn` / `maxconn` parameters (defaults typically 1–5).
- Supports **external pool injection**: callers can pass a pre-built `connection_pool` object (e.g., to share a pool across multiple `Memory` instances), overriding internal pool creation.
- Connection string or individual host/port/user/password parameters are also supported.

#### MySQL / azure_mysql (`mem0/vector_stores/azure_mysql.py`)
- Uses `dbutils.pooled_db.PooledDB` for connection pooling with configurable `minconn`/`maxconn`.
- Also supports external pool injection via `connection_pool` parameter.

#### OpenSearch (`mem0/configs/vector_stores/opensearch.py`)
- Configures `pool_maxsize` (default 20) for the underlying HTTP connection pool.

#### Weaviate (`mem0/vector_stores/weaviate.py`)
- Sets `skip_init_checks=True` and a short `Timeout(init=2.0)` to avoid blocking on startup.
- No explicit connection pooling or pool-size controls exposed.

#### Redis / Qdrant / Pinecone / Others
- Rely on the underlying client library's default connection management (no explicit pool configuration in Mem0's code).

#### SQLite History DB (`mem0/memory/storage.py`)
- A single `sqlite3.connect(db_path, check_same_thread=False)` connection is opened on construction and shared across threads via a `threading.Lock`.
- `check_same_thread=False` allows cross-thread use but SQLite's single-writer model means heavy concurrent writes could serialize or cause contention.
- No connection pooling (SQLite does not support it), and no reconnection logic if the connection drops.

#### HTTP API Client (`mem0/client/main.py`)
- Uses `httpx.AsyncClient` with a fixed `timeout=300` seconds for async operations; sync variant similarly configured.
- HTTP connection reuse is handled by `httpx`'s built-in connection pool.

---

### 2. Resource Cleanup

#### Synchronous `Memory.reset()` (`mem0/memory/main.py`)
- Explicitly closes the SQLite connection: `self.db.connection.execute("DROP TABLE IF EXISTS history"); self.db.connection.close()` then re-creates a fresh `SQLiteManager`.
- Calls `self.vector_store.delete_col()` to wipe data, then re-creates the store via `VectorStoreFactory`.
- **Gap**: Does **not** explicitly close vector store connections (e.g., PostgreSQL pool, Weaviate client) — only data is deleted, not connections closed.
- Does **not** clean up `self.graph` (graph DB) connections or `self.llm` client objects.

#### Asynchronous `AsyncMemory.reset()` (`mem0/memory/main.py`)
- More thorough: calls `gc.collect()` explicitly, then checks `hasattr(self.vector_store, "client")` and `hasattr(self.vector_store.client, "close")` before calling `self.vector_store.client.close()` (via `asyncio.to_thread`).
- Also closes the SQLite connection asynchronously before recreating it.
- Still does **not** close graph or LLM client resources.

#### HTTP API Client (`mem0/client/main.py`)
- `AsyncMem0Client` implements `__aenter__` / `__aexit__` (async context manager): on exit, calls `await self.async_client.aclose()`.
- **Gap**: The **synchronous** `MemoryClient` has **no** `__enter__`/`__exit__` context manager and no explicit `close()` method. The `httpx` sync client is not explicitly closed.

#### Vector Store Implementations
- None of the vector store classes (`pgvector.py`, `redis.py`, `weaviate.py`, etc.) implement `__del__`, `close()`, or context manager (`__enter__`/`__exit__`) methods.
- Connection pools (PostgreSQL, MySQL) are never explicitly shut down; they will be garbage-collected when the object is destroyed, relying on the underlying libraries' finalizers.
- No `atexit` hooks are registered anywhere in the codebase.

---

### 3. Reconnection & Retry Logic

#### HTTP API Client (`mem0/client/utils.py`)
- The `@api_error_handler` decorator (applied to all client API calls) catches `httpx.TimeoutException`, `httpx.ConnectError`, and rate-limit errors.
- On `429 Too Many Requests`, it reads `Retry-After` header and sleeps accordingly.
- On `TimeoutException` / `ConnectError`, it raises a structured `ConnectionError` with a user-facing suggestion ("check your internet connection") — **no automatic retry**.

#### AWS Bedrock LLM (`mem0/llms/aws_bedrock.py`)
- Calls `self._test_connection()` during initialization to verify credentials/connectivity (fail-fast pattern).
- No retry logic if the connection test fails or if calls fail at runtime.

#### Vector Stores & Graph DBs
- No reconnection logic anywhere. If a database connection is lost mid-operation, the exception will propagate to the caller unhandled.

---

### 4. Thread Pool Executors

- In `Memory` (sync class), parallel operations (add, get_all, search) spin up a `concurrent.futures.ThreadPoolExecutor` **on every call** using a `with` block, meaning a new thread pool is created and torn down for every single operation. This is functional but adds overhead.
- In `AsyncMemory`, operations are offloaded to threads via `asyncio.to_thread(...)` rather than a persistent executor.

---

### 5. Notable Gaps & Risks

| Concern | Status |
|---|---|
| `Memory` class context manager (`with Memory() as m`) | ❌ Not implemented |
| `AsyncMemory` class context manager | ❌ Not implemented |
| Vector store `close()` / `__del__` methods | ❌ Missing across all implementations |
| Sync `MemoryClient` explicit close | ❌ Missing |
| Automatic reconnection on dropped DB connections | ❌ Not present |
| Graph DB (`neo4j`, `kuzu`, `memgraph`) connection cleanup | ❌ Not explicit |
| SQLite concurrent write safety | ⚠️ Lock-based, single connection, no WAL mode |
| New `ThreadPoolExecutor` per operation | ⚠️ Inefficient; no persistent executor |
| `atexit` / signal handler for graceful shutdown | ❌ Not present |
| Async `AsyncMem0Client` context manager | ✅ Implemented |
| Async `Memory.reset()` vector store client close | ✅ Implemented (best-effort via `hasattr` check) |
| PostgreSQL / MySQL connection pooling | ✅ Well implemented with external injection support |
| HTTP client timeout configuration | ✅ 300-second timeout set |

---

### Summary

Mem0's lifecycle management is **partial and inconsistent**. The strongest patterns are in the database-backed vector stores (pgvector, azure_mysql) where connection pooling with configurable sizes and external pool injection are well-supported. The async `Memory.reset()` and the `AsyncMem0Client` context manager provide reasonable cleanup paths. However, there is no graceful shutdown story for the `Memory` class itself, no context manager support, no automatic reconnection anywhere, and vector store connection pools are never explicitly closed. Applications using Mem0 in long-running services should implement their own shutdown hooks to call `reset()` or close underlying client objects, and should be aware that connection pool teardown relies entirely on garbage collection.


## Q11: How are vector store backends tested? Is there a shared test structure, or are tests independent per backend?

## Vector Store Backend Testing in Mem0

### Structure Overview

There are **22 independent test files** in `tests/vector_stores/`, one per backend (Qdrant, Pinecone, Chroma, Milvus, Elasticsearch, Weaviate, FAISS, MongoDB, pgvector, Cassandra, Redis/Valkey, OpenSearch, Supabase, Upstash, Baidu, Databricks, LangChain, Azure AI Search, Azure MySQL, Neptune Analytics, Vertex AI, S3 Vectors).

### No Shared Test Infrastructure

- **No `conftest.py`** exists anywhere in the project — there are no shared pytest fixtures or test configuration files.
- **No shared base test class** for vector stores — each test file is entirely self-contained.
- The `VectorStoreBase` abstract class in `mem0/vector_stores/base.py` defines the interface (all backends implement the same abstract methods), but there is no corresponding abstract or shared *test* base class.

### Tests Are Independent Per Backend

Each test file independently:
1. Imports its specific backend class (e.g., `from mem0.vector_stores.qdrant import Qdrant`)
2. Defines its own fixtures/setup to mock that backend's specific client library
3. Mocks all external dependencies using `unittest.mock` (MagicMock, Mock, patch) to avoid real network/database connections
4. Tests the backend's implementation in isolation

### Two Testing Frameworks Used

Most backends use **pytest** with `@pytest.fixture` decorators:
- Chroma, Pinecone, Milvus, MongoDB, FAISS, Cassandra, Baidu, Databricks, pgvector, etc.

A few backends use **`unittest.TestCase`** with `setUp`/`setUpClass`:
- Elasticsearch (`TestElasticsearchDB(unittest.TestCase)`)
- Weaviate (`TestWeaviateDB(unittest.TestCase)`)
- Qdrant (`TestQdrant(unittest.TestCase)`)

Milvus uses a hybrid approach: a `TestMilvusDB` class but with pytest-style fixtures inside it.

### Common Operations Tested

Despite independence, tests naturally cover the same abstract interface defined in `VectorStoreBase`:

| Operation | Description |
|-----------|-------------|
| `create_col` | Create a collection/index |
| `insert` | Upsert vectors with payloads and IDs |
| `search` | Query by vector similarity, with optional filters |
| `delete` | Remove a vector by ID |
| `update` | Update a vector's embedding or payload |
| `get` | Retrieve a single vector by ID |
| `list` | List all stored memories (with filters) |
| `list_cols` | List all collections |
| `delete_col` | Delete a collection |
| `col_info` | Get collection metadata |
| `reset` | Delete and recreate a collection |

Beyond these core operations, each file also tests backend-specific behavior:
- **Filter construction** (Milvus, Chroma, MongoDB — building filter expressions from `user_id`, `agent_id`, `run_id`)
- **Authentication variants** (Azure AI Search — API key vs. DefaultAzureCredential)
- **Config validation** (Azure AI Search — compression type enums)
- **Index/collection creation logic** (Pinecone — skips creation if index already exists)
- **Batch operations** (Milvus — verifies batch insert rather than per-item loop)

### Summary

The tests follow a **shared-by-convention, independent-by-implementation** pattern. All backends implement the same `VectorStoreBase` interface, so naturally they test the same set of operations — but there is no actual shared test code, no common fixtures, and no parametrized test suite that runs across all backends. Each backend's test file is a standalone unit test module using mocked clients specific to that library.

## Q12: What mocking or fixture patterns are used to isolate tests from real database connections?

## Mocking and Fixture Patterns Used to Isolate Tests from Real Database Connections

The Mem0 test suite uses a rich variety of mocking and fixture patterns to prevent tests from hitting real databases. Here is a comprehensive breakdown:

---

### 1. Direct Mock Client Injection via Constructor

The most common pattern: a `MagicMock` (often spec-constrained) is created and passed directly to the class under test as the `client` parameter, bypassing any real connection.

**Files:** `tests/vector_stores/test_qdrant.py`, `tests/vector_stores/test_pinecone.py`, `tests/vector_stores/test_weaviate.py`

```python
# test_qdrant.py
class TestQdrant(unittest.TestCase):
    def setUp(self):
        self.client_mock = MagicMock(spec=QdrantClient)
        self.qdrant = Qdrant(
            collection_name="test_collection",
            embedding_model_dims=128,
            client=self.client_mock,  # injected mock
            ...
        )
```

Using `spec=QdrantClient` (or `spec=weaviate.WeaviateClient`) restricts the mock to only allow attributes/methods that the real class has, catching misuse early.

---

### 2. `patch` as Context Manager Inside `@pytest.fixture`

A `@pytest.fixture` uses `with patch("module.ClassName") as mock_class:` to intercept construction of the real client during the fixture's setup, then yields the instance.

**Files:** `tests/vector_stores/test_chroma.py`, `tests/vector_stores/test_cassandra.py`, `tests/vector_stores/test_milvus.py`, `tests/vector_stores/test_mongodb.py`

```python
# test_chroma.py
@pytest.fixture
def mock_chromadb_client():
    with patch("chromadb.Client") as mock_client:
        yield mock_client

@pytest.fixture
def chromadb_instance(mock_chromadb_client):
    mock_collection = Mock()
    mock_chromadb_client.return_value.get_or_create_collection.return_value = mock_collection
    return ChromaDB(collection_name="test_collection", client=mock_chromadb_client.return_value)

# test_cassandra.py
@pytest.fixture
def cassandra_instance(mock_cluster, mock_session):
    with patch('mem0.vector_stores.cassandra.Cluster') as mock_cluster_class:
        mock_cluster_class.return_value = mock_cluster
        instance = CassandraDB(contact_points=['127.0.0.1'], ...)
        yield instance
```

Chained fixtures (e.g., `cassandra_instance` depending on `mock_cluster` which depends on `mock_session`) build up a complete mock dependency graph.

---

### 3. `@patch` Decorator on Tests or Fixtures (for `unittest.TestCase`)

Used in `unittest.TestCase`-style tests, `@patch(...)` is applied directly to test methods or `setUp`, intercepting module-level imports of database client constructors.

**Files:** `tests/vector_stores/test_pgvector.py`, `tests/vector_stores/test_mongodb.py`

```python
# test_pgvector.py
@patch('mem0.vector_stores.pgvector.PSYCOPG_VERSION', 3)
@patch('mem0.vector_stores.pgvector.ConnectionPool')
def test_init_with_individual_params_psycopg3(self, mock_psycopg_pool):
    mock_psycopg_pool.return_value = self.mock_pool_psycopg
    pgvector = PGVector(dbname="test_db", ...)
    mock_psycopg_pool.assert_called_once_with(...)

# test_mongodb.py
@pytest.fixture
@patch("mem0.vector_stores.mongodb.MongoClient")
def mongo_vector_fixture(mock_mongo_client):
    mock_client = mock_mongo_client.return_value
    ...
    mongo_vector = MongoDB(..., mongo_uri="mongodb://username:password@localhost:27017")
    return mongo_vector, mock_collection, mock_db
```

---

### 4. `patcher.start()` / `patcher.stop()` with `addCleanup` (in `setUp`/`tearDown`)

For `unittest.TestCase` classes that need patcher lifecycle management, `patch(...)` is stored, started in `setUp`, and cleaned up either via `addCleanup(patcher.stop)` or a `tearDown` method.

**Files:** `tests/vector_stores/test_elasticsearch.py`, `tests/vector_stores/test_weaviate.py`, `tests/memory/test_neptune_memory.py`

```python
# test_elasticsearch.py
def setUp(self):
    self.client_mock = MagicMock(spec=Elasticsearch)
    patcher = patch("mem0.vector_stores.elasticsearch.Elasticsearch",
                    return_value=self.client_mock)
    self.mock_es = patcher.start()
    self.addCleanup(patcher.stop)  # guaranteed cleanup
    self.es_db = ElasticsearchDB(...)

# test_neptune_memory.py
def setUp(self):
    self.neptune_graph_patcher = patch("mem0.graphs.neptune.neptunedb.NeptuneGraph")
    self.mock_neptune_graph = self.neptune_graph_patcher.start()
    self.mock_neptune_graph.return_value = self.mock_graph

def tearDown(self):
    self.neptune_graph_patcher.stop()
```

---

### 5. `patch.object` for Patching Static/Class Methods

Used when a specific method on a class (often a static or class method) needs to be replaced rather than the entire class constructor.

**Files:** `tests/memory/test_neptune_memory.py`

```python
self.create_embedding_model_patcher = patch.object(
    NeptuneBase, "_create_embedding_model"
)
self.mock_create_embedding_model = self.create_embedding_model_patcher.start()
self.mock_create_embedding_model.return_value = self.mock_embedding_model
```

---

### 6. Factory Mocking via `pytest-mock`'s `mocker` Fixture

The high-level `Memory` class creates its dependencies (LLM, embedder, vector store) through factory classes. Tests use `mocker.patch(...)` to intercept all three factory calls, completely decoupling the `Memory` object from any real backend.

**Files:** `tests/memory/test_main.py`

```python
def _setup_mocks(mocker):
    mock_embedder = mocker.MagicMock()
    mock_embedder.return_value.embed.return_value = [0.1, 0.2, 0.3]
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mock_embedder)

    mock_vector_store = mocker.MagicMock()
    mock_vector_store.return_value.search.return_value = []
    mocker.patch(
        "mem0.utils.factory.VectorStoreFactory.create",
        side_effect=[mock_vector_store.return_value, mocker.MagicMock()]
    )

    mock_llm = mocker.MagicMock()
    mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm)
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())

    return mock_llm, mock_vector_store
```

`mocker.patch` (from `pytest-mock`) automatically undoes all patches after the test, so no manual cleanup is needed.

---

### 7. In-Memory SQLite Database (`:memory:`)

For tests of `SQLiteManager` (the history storage layer) that need a real database but want isolation, the special SQLite `:memory:` path creates a fully functional in-memory database with no file I/O and no persistence.

**Files:** `tests/memory/test_storage.py`, `tests/memory/test_kuzu.py`

```python
# test_storage.py
@pytest.fixture
def memory_manager(self):
    manager = SQLiteManager(":memory:")
    yield manager
    if manager.connection:
        manager.close()

# test_kuzu.py (inside mock config)
config.graph_store.config.db = ":memory:"
```

---

### 8. Temporary File Fixture for File-Based Databases

For tests requiring an actual file (e.g., to test file creation/deletion), `tempfile.NamedTemporaryFile` creates a unique disposable file per test, yielded and cleaned up after use.

**Files:** `tests/memory/test_storage.py`

```python
@pytest.fixture
def temp_db_path(self):
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()
    yield temp_db.name
    if os.path.exists(temp_db.name):
        os.unlink(temp_db.name)
```

---

### 9. `patch.dict('sys.modules', ...)` for Optional Import Mocking

When a module may not be installed (e.g., `langchain_neo4j`), `patch.dict` on `sys.modules` injects a `Mock()` so the import succeeds without the real package being present.

**Files:** `tests/memory/test_neo4j_cypher_syntax.py`

```python
with patch.dict('sys.modules', {'langchain_neo4j': Mock()}):
    from mem0.memory.graph_memory import MemoryGraph
```

---

### 10. Environment Variable Patching in `setUpClass`

Some test classes for cloud-connected databases (Elasticsearch, Weaviate) save original env vars, override them with dummy values for the test run, and restore them in `tearDownClass`.

**Files:** `tests/vector_stores/test_elasticsearch.py`, `tests/vector_stores/test_weaviate.py`

```python
@classmethod
def setUpClass(cls):
    cls.original_env = {"ES_URL": os.getenv("ES_URL", "http://localhost:9200"), ...}
    os.environ["ES_URL"] = "http://localhost:9200"
    os.environ["ES_USERNAME"] = "test_user"
    ...
```

---

### 11. `side_effect` for Sequential / Error-Triggering Mock Responses

`side_effect` is used on mock return values to simulate different behaviors on successive calls (e.g., first call succeeds, second fails) or to raise exceptions.

**Files:** `tests/memory/test_main.py`

```python
mock_memory.llm.generate_response.side_effect = [
    '{"facts": ["test fact"]}',  # first call: valid JSON
    ""                            # second call: empty (triggers error handling)
]
```

---

### Summary Table

| Pattern | Tool | Isolation Target |
|---|---|---|
| Direct mock client injection | `MagicMock(spec=...)` | DB client construction |
| `patch` context manager in fixture | `unittest.mock.patch` | Client class import |
| `@patch` decorator | `unittest.mock.patch` | Class constructor in test |
| `patcher.start()`/`addCleanup(stop)` | `unittest.mock.patch` | Lifecycle-managed patches |
| `patch.object` | `unittest.mock.patch` | Specific class methods |
| Factory mocking via `mocker` | `pytest-mock` | Full dependency graph |
| In-memory SQLite | `SQLiteManager(":memory:")` | File I/O for storage |
| Temp file fixture | `tempfile` + `yield` | File system state |
| `sys.modules` patching | `patch.dict` | Optional imports |
| Env var patching | `os.environ` | Credentials/config |
| `side_effect` | `MagicMock.side_effect` | Multi-step call sequences |


## Q13: Based on everything we've discussed, produce a step-by-step integration guide for adding a new vector store backend to this project. Cover every file to create, every file to modify, every registration point, config class, and test requirement. The guide should be self-contained.


# Step-by-Step Integration Guide: Adding a New Vector Store Backend to Mem0

This guide uses **`mystore`** as the example provider name (replace it with your actual name, e.g., `weaviate`, `lancedb`, etc.).

---

## Overview of Required Changes

| Action | File |
|--------|------|
| **CREATE** | `mem0/configs/vector_stores/mystore.py` |
| **CREATE** | `mem0/vector_stores/mystore.py` |
| **CREATE** | `tests/vector_stores/test_mystore.py` |
| **MODIFY** | `mem0/vector_stores/configs.py` |
| **MODIFY** | `mem0/utils/factory.py` |

No other files need to change (the `__init__.py` files in both `mem0/configs/vector_stores/` and `mem0/vector_stores/` are intentionally empty and rely on dynamic imports).

---

## Step 1 — Create the Config Class

**File to create:** `mem0/configs/vector_stores/mystore.py`

This file holds a Pydantic `BaseModel` describing every configuration option for the new backend.

```python
# mem0/configs/vector_stores/mystore.py

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class MyStoreConfig(BaseModel):
    """Configuration for the MyStore vector store backend."""

    # Connection details — adapt to what your backend requires
    host: Optional[str] = Field(None, description="MyStore server hostname")
    port: Optional[int] = Field(8080, description="MyStore server port")
    api_key: Optional[str] = Field(None, description="API key for MyStore cloud")
    url: Optional[str] = Field(None, description="Full URL, e.g. https://my-instance.mystore.io")

    # Collection / index settings
    collection_name: str = Field("mem0", description="Name of the collection / index to use")
    embedding_model_dims: int = Field(1536, description="Dimensionality of the embedding vectors")
    distance: str = Field("cosine", description="Distance metric: 'cosine', 'euclidean', 'dotproduct'")

    # Optional local-storage path (used by VectorStoreConfig to auto-set a default)
    path: Optional[str] = Field(None, description="Local path for file-based backends")

    @model_validator(mode="after")
    def validate_config(self) -> "MyStoreConfig":
        # Enforce that at least one connection method is specified.
        # VectorStoreConfig will already have set `path` to `/tmp/mystore` as a
        # fallback if `path` is declared in annotations and nothing else is given.
        if not any([self.url, self.host, self.path]):
            self.path = "/tmp/mystore"
        return self
```

**Rules:**
- The class name must match the value you will register in `VectorStoreConfig._provider_configs` (next step).
- Always include `collection_name` and `embedding_model_dims`; these are consumed by `VectorStoreFactory.create()` which calls `config.model_dump()`.
- Declare `path` as a field (even if unused) if you want `VectorStoreConfig` to automatically inject a `/tmp/<provider>` default when no path is given.

---

## Step 2 — Register the Config in `VectorStoreConfig`

**File to modify:** `mem0/vector_stores/configs.py`

Add one entry to `_provider_configs` and one entry to the auto-path logic (already handled automatically if `path` is in the model annotations).

```python
# mem0/vector_stores/configs.py  — excerpt showing the diff

_provider_configs: Dict[str, str] = {
    "qdrant": "QdrantConfig",
    "chroma": "ChromaDbConfig",
    # ... existing entries ...
    "s3_vectors": "S3VectorsConfig",
    # ↓ ADD THIS LINE ↓
    "mystore": "MyStoreConfig",
}
```

The value (`"MyStoreConfig"`) must exactly match the class name defined in Step 1.  
The key (`"mystore"`) becomes the string users pass as `provider` in their `MemoryConfig`.

The `model_validator` in `VectorStoreConfig` uses this mapping to dynamically import  
`mem0.configs.vector_stores.<provider>` and instantiate the appropriate config class — no further changes are needed in this file.

---

## Step 3 — Register the Implementation in `VectorStoreFactory`

**File to modify:** `mem0/utils/factory.py`

Add one entry to `VectorStoreFactory.provider_to_class`:

```python
# mem0/utils/factory.py  — excerpt showing the diff

class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
        "chroma": "mem0.vector_stores.chroma.ChromaDB",
        # ... existing entries ...
        "s3_vectors": "mem0.vector_stores.s3_vectors.S3Vectors",
        # ↓ ADD THIS LINE ↓
        "mystore": "mem0.vector_stores.mystore.MyStore",
    }
```

The value is the fully-qualified dotted path `mem0.vector_stores.<module>.<ClassName>`.  
`load_class()` splits on the last `.` and uses `importlib.import_module` — the backend library is therefore imported lazily (only when this provider is actually used).

---

## Step 4 — Create the Implementation Class

**File to create:** `mem0/vector_stores/mystore.py`

Extend `VectorStoreBase` and implement all 11 abstract methods. The constructor receives `**kwargs`; the factory calls `config.model_dump()` and passes the result as keyword arguments.

```python
# mem0/vector_stores/mystore.py

import logging
from typing import Optional

try:
    from mystore_client import MyStoreClient  # replace with real SDK import
except ImportError:
    raise ImportError(
        "The 'mystore-client' library is required. "
        "Install it with 'pip install mystore-client'."
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData:
    """Uniform result object returned by search/get/list."""
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class MyStore(VectorStoreBase):
    """Mem0 vector-store backend for MyStore."""

    def __init__(self, **kwargs):
        # The factory calls model_dump() on the config and spreads it as kwargs.
        # Pull out the fields you need:
        self.collection_name = kwargs.get("collection_name", "mem0")
        self.embedding_model_dims = kwargs.get("embedding_model_dims", 1536)
        self.distance = kwargs.get("distance", "cosine")

        url = kwargs.get("url")
        api_key = kwargs.get("api_key")
        host = kwargs.get("host")
        port = kwargs.get("port", 8080)
        path = kwargs.get("path")

        # Initialise the client
        if url and api_key:
            self.client = MyStoreClient(url=url, api_key=api_key)
        elif host:
            self.client = MyStoreClient(host=host, port=port)
        elif path:
            self.client = MyStoreClient(path=path)
        else:
            raise ValueError("MyStore: no valid connection configuration provided.")

        # Ensure the collection exists on startup
        self.create_col(
            name=self.collection_name,
            vector_size=self.embedding_model_dims,
            distance=self.distance,
        )

    # ------------------------------------------------------------------ #
    #  Required abstract methods (must all be implemented)                #
    # ------------------------------------------------------------------ #

    def create_col(self, name, vector_size, distance):
        """Create a new collection / index. Skip if it already exists."""
        if self.client.collection_exists(name):
            return
        self.client.create_collection(name=name, dims=vector_size, metric=distance)

    def insert(self, vectors, payloads=None, ids=None):
        """Upsert vectors with associated payloads and IDs."""
        records = [
            {"id": id_, "vector": vec, "payload": payload}
            for id_, vec, payload in zip(ids, vectors, payloads or [{}] * len(vectors))
        ]
        self.client.upsert(collection=self.collection_name, records=records)

    def search(self, query, vectors, limit=5, filters=None):
        """Return the top-`limit` nearest neighbours to `vectors`.

        Args:
            query: The raw query string (provided for logging/debugging; not always needed).
            vectors: The query embedding as a list of floats.
            limit: Maximum number of results.
            filters: Dict of metadata key/value pairs to pre-filter by.

        Returns:
            List[OutputData]
        """
        query_filter = self._build_filter(filters)
        hits = self.client.search(
            collection=self.collection_name,
            vector=vectors,
            limit=limit,
            filter=query_filter,
        )
        return [OutputData(id=h.id, score=h.score, payload=h.payload) for h in hits]

    def delete(self, vector_id):
        """Delete a single record by ID."""
        self.client.delete(collection=self.collection_name, ids=[vector_id])

    def update(self, vector_id, vector=None, payload=None):
        """Update vector and/or payload for an existing record."""
        if vector is not None:
            self.client.upsert(
                collection=self.collection_name,
                records=[{"id": vector_id, "vector": vector, "payload": payload}],
            )
        else:
            self.client.update_payload(
                collection=self.collection_name,
                id=vector_id,
                payload=payload,
            )

    def get(self, vector_id):
        """Retrieve a single record by ID. Returns OutputData or None."""
        result = self.client.get(
            collection=self.collection_name,
            ids=[vector_id],
            with_vectors=True,
        )
        if result:
            r = result[0]
            return OutputData(id=r.id, score=None, payload=r.payload)
        return None

    def list_cols(self):
        """Return all collections managed by this client."""
        return self.client.list_collections()

    def delete_col(self):
        """Drop the current collection entirely."""
        self.client.delete_collection(self.collection_name)

    def col_info(self):
        """Return metadata / stats about the current collection."""
        return self.client.collection_info(self.collection_name)

    def list(self, filters=None, limit=None):
        """Return all records in the collection, optionally filtered."""
        query_filter = self._build_filter(filters)
        results = self.client.scroll(
            collection=self.collection_name,
            filter=query_filter,
            limit=limit,
        )
        return [OutputData(id=r.id, score=None, payload=r.payload) for r in results]

    def reset(self):
        """Drop and recreate the collection (used during testing / reset flows)."""
        self.delete_col()
        self.create_col(
            name=self.collection_name,
            vector_size=self.embedding_model_dims,
            distance=self.distance,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_filter(self, filters):
        """Convert a plain dict of key/value pairs to the backend's filter object."""
        if not filters:
            return None
        # Adapt to your backend's filter API:
        return {"must": [{"key": k, "match": {"value": v}} for k, v in filters.items()]}
```

**Key contracts expected by the rest of Mem0:**

| Method | What callers expect |
|--------|---------------------|
| `insert` | Accepts parallel lists `ids`, `vectors`, `payloads` |
| `search` | Returns a list of objects with `.id`, `.score`, `.payload` attributes |
| `get` | Returns a single such object or `None` |
| `list` | Returns a list of such objects (score may be `None`) |
| `reset` | Wipes all data — used in tests and the `reset()` API |

---

## Step 5 — Write the Tests

**File to create:** `tests/vector_stores/test_mystore.py`

Follow the existing pattern: mock the third-party client, instantiate the class directly, and assert every public method.

```python
# tests/vector_stores/test_mystore.py

import unittest
import uuid
from unittest.mock import MagicMock, patch

from mem0.vector_stores.mystore import MyStore


class TestMyStore(unittest.TestCase):

    def setUp(self):
        # Patch the external SDK so no real network call is made
        patcher = patch("mem0.vector_stores.mystore.MyStoreClient")
        self.mock_client_cls = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_client = MagicMock()
        self.mock_client_cls.return_value = self.mock_client

        # collection_exists → False so create_col actually calls create_collection
        self.mock_client.collection_exists.return_value = False

        self.store = MyStore(
            host="localhost",
            port=8080,
            collection_name="test_col",
            embedding_model_dims=128,
            distance="cosine",
        )

    # --- create_col -------------------------------------------------------

    def test_create_col_when_not_exists(self):
        self.mock_client.collection_exists.return_value = False
        self.store.create_col("new_col", 128, "cosine")
        self.mock_client.create_collection.assert_called_with(
            name="new_col", dims=128, metric="cosine"
        )

    def test_create_col_skips_when_exists(self):
        self.mock_client.collection_exists.return_value = True
        self.store.create_col("test_col", 128, "cosine")
        # create_collection should NOT be called again
        self.mock_client.create_collection.assert_not_called()

    # --- insert -----------------------------------------------------------

    def test_insert(self):
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        vectors = [[0.1] * 128, [0.2] * 128]
        payloads = [{"user_id": "alice"}, {"user_id": "bob"}]

        self.store.insert(vectors=vectors, payloads=payloads, ids=ids)

        self.mock_client.upsert.assert_called_once()
        call_kwargs = self.mock_client.upsert.call_args[1]
        self.assertEqual(len(call_kwargs["records"]), 2)

    # --- search -----------------------------------------------------------

    def test_search_without_filter(self):
        mock_hit = MagicMock(id=str(uuid.uuid4()), score=0.95, payload={"user_id": "alice"})
        self.mock_client.search.return_value = [mock_hit]

        results = self.store.search(query="test", vectors=[0.1] * 128, limit=5)

        self.mock_client.search.assert_called_once_with(
            collection="test_col",
            vector=[0.1] * 128,
            limit=5,
            filter=None,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 0.95)

    def test_search_with_filter(self):
        self.mock_client.search.return_value = []
        self.store.search(query="q", vectors=[0.1] * 128, filters={"user_id": "alice"})

        call_kwargs = self.mock_client.search.call_args[1]
        self.assertIsNotNone(call_kwargs["filter"])

    # --- delete -----------------------------------------------------------

    def test_delete(self):
        vid = str(uuid.uuid4())
        self.store.delete(vid)
        self.mock_client.delete.assert_called_once_with(
            collection="test_col", ids=[vid]
        )

    # --- update -----------------------------------------------------------

    def test_update_with_vector(self):
        vid = str(uuid.uuid4())
        self.store.update(vid, vector=[0.3] * 128, payload={"data": "new"})
        self.mock_client.upsert.assert_called()

    def test_update_payload_only(self):
        vid = str(uuid.uuid4())
        self.store.update(vid, payload={"data": "updated"})
        self.mock_client.update_payload.assert_called_once_with(
            collection="test_col", id=vid, payload={"data": "updated"}
        )

    # --- get --------------------------------------------------------------

    def test_get_existing(self):
        mock_rec = MagicMock(id="abc", payload={"key": "val"})
        self.mock_client.get.return_value = [mock_rec]
        result = self.store.get("abc")
        self.assertEqual(result.id, "abc")

    def test_get_missing(self):
        self.mock_client.get.return_value = []
        result = self.store.get("nonexistent")
        self.assertIsNone(result)

    # --- list / col_info / list_cols / delete_col -------------------------

    def test_list(self):
        self.mock_client.scroll.return_value = []
        results = self.store.list()
        self.assertEqual(results, [])

    def test_list_cols(self):
        self.store.list_cols()
        self.mock_client.list_collections.assert_called_once()

    def test_delete_col(self):
        self.store.delete_col()
        self.mock_client.delete_collection.assert_called_once_with("test_col")

    def test_col_info(self):
        self.store.col_info()
        self.mock_client.collection_info.assert_called_once_with("test_col")

    # --- reset ------------------------------------------------------------

    def test_reset(self):
        self.mock_client.collection_exists.return_value = False
        self.store.reset()
        self.mock_client.delete_collection.assert_called_once_with("test_col")
        self.mock_client.create_collection.assert_called()


if __name__ == "__main__":
    unittest.main()
```

---

## Summary Checklist

```
□ 1. CREATE  mem0/configs/vector_stores/mystore.py
       - Pydantic BaseModel named MyStoreConfig
       - Fields: collection_name, embedding_model_dims, distance, connection params, optional path
       - model_validator to enforce at least one connection method

□ 2. MODIFY  mem0/vector_stores/configs.py
       - Add "mystore": "MyStoreConfig"  to  VectorStoreConfig._provider_configs

□ 3. MODIFY  mem0/utils/factory.py
       - Add "mystore": "mem0.vector_stores.mystore.MyStore"
         to  VectorStoreFactory.provider_to_class

□ 4. CREATE  mem0/vector_stores/mystore.py
       - class MyStore(VectorStoreBase)
       - __init__ accepts **kwargs (factory passes model_dump() output)
       - Implement ALL 11 abstract methods:
           create_col, insert, search, delete, update,
           get, list_cols, delete_col, col_info, list, reset
       - Return OutputData(id, score, payload) objects from search/get/list
       - Guard SDK import with try/except ImportError

□ 5. CREATE  tests/vector_stores/test_mystore.py
       - Mock the third-party client
       - Test every public method (happy path + edge cases)
       - Verify filter construction
       - Verify reset calls delete_col then create_col
```

## Using the New Backend

Once all five steps are complete, users can enable the backend via:

```python
from mem0 import Memory

m = Memory.from_config({
    "vector_store": {
        "provider": "mystore",
        "config": {
            "host": "localhost",
            "port": 8080,
            "collection_name": "my_memories",
            "embedding_model_dims": 1536,
        }
    }
})
```

Or programmatically:

```python
from mem0.vector_stores.configs import VectorStoreConfig

cfg = VectorStoreConfig(
    provider="mystore",
    config={"host": "localhost", "port": 8080}
)
```


## Q14: Using the integration guide you just produced, walk through the existing Qdrant implementation step by step. For each step in the guide, verify it matches what Qdrant actually did. Flag any steps that are missing, wrong, or incomplete. If Qdrant does something the guide doesn't cover, that's a gap in the guide.

## Integration Guide Verification: Walking Through the Qdrant Implementation

Below I verify every step of the integration guide (from Q13/A13) against what Qdrant actually did, flagging gaps, deviations, and undocumented behaviors.

---

### Guide Overview Table vs. Reality

The guide prescribes exactly **5 files**:

| Action | File (Guide) | Qdrant Reality |
|--------|-------------|----------------|
| CREATE | `mem0/configs/vector_stores/mystore.py` | ✅ `mem0/configs/vector_stores/qdrant.py` exists |
| CREATE | `mem0/vector_stores/mystore.py` | ✅ `mem0/vector_stores/qdrant.py` exists |
| CREATE | `tests/vector_stores/test_mystore.py` | ✅ `tests/vector_stores/test_qdrant.py` exists |
| MODIFY | `mem0/vector_stores/configs.py` | ✅ `qdrant` is registered |
| MODIFY | `mem0/utils/factory.py` | ✅ `qdrant` is registered |

All five files exist. ✅

---

## Step 1 — Config Class (`mem0/configs/vector_stores/qdrant.py`)

**Guide says:** Create a Pydantic `BaseModel` with connection fields (`host`, `port`, `api_key`, `url`, `path`), `collection_name`, `embedding_model_dims`, `distance`, and a `@model_validator` that enforces at least one connection method.

**What Qdrant actually does:**

```python
class QdrantConfig(BaseModel):
    collection_name: str = "mem0"
    embedding_model_dims: Optional[int] = 1536
    client: Optional[Any] = None       # ← NOT in guide
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    on_disk: bool = False              # ← NOT in guide

    @model_validator(mode="before")
    @classmethod
    def check_host_port_or_path(cls, values): ...

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values): ...
```

**Verification:**

| Guide Element | Qdrant Reality | Status |
|---|---|---|
| `collection_name` field | ✅ Present, default `"mem0"` | Match |
| `embedding_model_dims` field | ✅ Present, default `1536` | Match |
| `host`, `port`, `api_key`, `url`, `path` fields | ✅ All present | Match |
| `distance` field | ❌ **Missing from `QdrantConfig`** | **GAP IN GUIDE** |
| `@model_validator` for connection validation | ✅ `check_host_port_or_path` does this | Match |
| `client` field (pre-built client injection) | ✅ Present in Qdrant | **GUIDE MISSES THIS** |
| `on_disk` field | ✅ Present in Qdrant | **GUIDE MISSES THIS** |
| Second validator `validate_extra_fields` | ✅ Rejects unknown fields | **GUIDE MISSES THIS** |
| Single `@model_validator(mode="after")` | Qdrant uses `mode="before"` with two separate validators | Minor discrepancy |

**Flags:**
- 🔴 **Guide omits `client` field**: Qdrant supports injecting a pre-built `QdrantClient` directly (for testing or custom setups). The guide never mentions this.
- 🔴 **Guide omits `on_disk` field**: Qdrant supports on-disk payload indexing. Not mentioned.
- 🔴 **Guide includes `distance` field**: Qdrant's config does NOT have a `distance` field — it is hardcoded in the implementation as `Distance.COSINE`. The guide is incorrect here.
- 🟡 **Guide uses single `mode="after"` validator**: Qdrant uses two `mode="before"` validators — one for connection logic, one to reject extra fields. Using "before" is safer and is what the real codebase does.
- 🔴 **Guide misses `validate_extra_fields`**: A second validator that raises `ValueError` on any unrecognized config key is missing from the guide entirely.

---

## Step 2 — Register in `VectorStoreConfig` (`mem0/vector_stores/configs.py`)

**Guide says:** Add `"mystore": "MyStoreConfig"` to `_provider_configs`. The `model_validator` auto-imports the config class.

**What Qdrant actually does:** `"qdrant": "QdrantConfig"` is registered in `_provider_configs`. The auto-path injection logic sets a `/tmp/qdrant` default when `path` is not provided. ✅

**Verification:**

| Guide Element | Qdrant Reality | Status |
|---|---|---|
| Single line in `_provider_configs` | ✅ Present | Match |
| Auto-path default set by `VectorStoreConfig` | ✅ Qdrant gets `/tmp/qdrant` as default path | Match |
| No other files need changes | ✅ Correct | Match |

**Flags:** None — this step is accurate. ✅

---

## Step 3 — Register in `VectorStoreFactory` (`mem0/utils/factory.py`)

**Guide says:** Add `"mystore": "mem0.vector_stores.mystore.MyStore"` to `VectorStoreFactory.provider_to_class`.

**What Qdrant actually does:** `"qdrant": "mem0.vector_stores.qdrant.Qdrant"` is present in the factory. ✅

**Verification:**

| Guide Element | Qdrant Reality | Status |
|---|---|---|
| Single entry in `provider_to_class` | ✅ Present | Match |
| Dotted module path format | ✅ `mem0.vector_stores.qdrant.Qdrant` | Match |
| Lazy import via `load_class()` | ✅ Correct | Match |

**Flags:** None — this step is accurate. ✅

---

## Step 4 — Implementation Class (`mem0/vector_stores/qdrant.py`)

**Guide says:** Extend `VectorStoreBase`, wrap the SDK import in `try/except ImportError`, implement all 11 abstract methods, and use an `OutputData` result class.

**What Qdrant actually does:**

```python
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import (Distance, PointStruct,
        VectorParams, Filter, FieldCondition, MatchValue,
        MatchAny, PayloadSchemaType)
except ImportError:
    raise ImportError("The 'qdrant-client' library is required...")

class Qdrant(VectorStoreBase):
    def __init__(self, collection_name, embedding_model_dims, client=None,
                 host=None, port=None, path=None, url=None,
                 api_key=None, on_disk=False):
        ...
```

**Verification of constructor:**

| Guide Element | Qdrant Reality | Status |
|---|---|---|
| `try/except ImportError` with pip hint | ✅ Present | Match |
| Extend `VectorStoreBase` | ✅ Present | Match |
| `**kwargs` constructor style | ❌ Qdrant uses **explicit named parameters**, not `**kwargs` | **GUIDE INACCURATE** |
| `client` injection support | ✅ Present (`if client: self.client = client`) | **GUIDE MISSES THIS** |
| `on_disk` parameter | ✅ Present | **GUIDE MISSES THIS** |
| Connection branching (url/host/path) | ✅ Present | Match |
| Call `create_col` at end of `__init__` | ✅ `self.create_col(embedding_model_dims, on_disk)` | Match (minor: guide passes `name=` explicitly, Qdrant uses `self.collection_name`) |

**Verification of all 11 methods:**

| Method | Guide Signature | Qdrant Reality | Status |
|---|---|---|---|
| `create_col(name, vector_size, distance)` | Creates with skip-if-exists | ✅ Creates `VectorParams(size=..., distance=Distance.COSINE)`, skips if exists. Uses hardcoded `Distance.COSINE` not a parameter. Also adds payload indexes for `user_id`, `run_id`, `agent_id`, `hash` | **GUIDE MISSES PAYLOAD INDEXES** |
| `insert(vectors, payloads, ids)` | Upserts records | ✅ Calls `upsert` with `PointStruct` objects | Match |
| `search(query, vectors, limit, filters)` | Returns `List[OutputData]` | ❌ Returns **native `PointStruct` objects** (not `OutputData`). Qdrant is the only backend that does NOT use `OutputData`. | **GUIDE WRONG** |
| `delete(vector_id)` | Deletes by ID | ✅ Uses `models.PointIdsList` | Match |
| `update(vector_id, vector, payload)` | Updates vector and/or payload | ✅ Handles both cases separately | Match |
| `get(vector_id)` | Returns `OutputData` or `None` | ❌ Returns native Qdrant result with `.id` and `.payload` (not `OutputData`) | **GUIDE WRONG** |
| `list_cols()` | Returns all collections | ✅ Present | Match |
| `delete_col()` | Deletes collection | ✅ Present | Match |
| `col_info()` | Returns collection info | ✅ Present | Match |
| `list(filters, limit)` | Returns all memories | ✅ Present, uses `scroll` API | Match |
| `reset()` | Deletes and recreates collection | ✅ Calls `delete_col()` then `create_col()` | Match |

**Additional method Qdrant has that the guide doesn't mention:**

- 🔴 **`_build_filters(filters)`**: A private helper method that converts the generic `filters` dict into Qdrant-native `Filter`/`FieldCondition`/`MatchValue`/`MatchAny` objects. This is substantial logic (~30 lines) that the guide completely ignores. The guide says nothing about implementing filter translation.

**Flags for Step 4:**
- 🔴 **`**kwargs` vs explicit params**: The guide says use `**kwargs` and call `kwargs.get(...)`. Qdrant uses explicit named parameters. Explicit params are safer (IDE support, validation, clearer errors).
- 🔴 **`OutputData` class**: The guide says to define and use an `OutputData` class. Qdrant does NOT — it returns native `PointStruct` objects directly. This is an inconsistency the guide should acknowledge (most backends use `OutputData`, Qdrant is the exception).
- 🔴 **Filter translation helper missing**: Every backend that supports filtering must implement filter translation from the generic dict format to the backend's native filter API. The guide doesn't mention this at all.
- 🔴 **Payload indexing in `create_col`**: Qdrant creates payload field indexes for `user_id`, `run_id`, `agent_id`, and `hash` after creating the collection. This is essential for filtering performance and correctness. The guide never mentions adding indexes.
- 🔴 **`distance` hardcoded**: The guide includes `distance` as a constructor param passed through. Qdrant hardcodes `Distance.COSINE`. A real implementation should document this choice.
- 🟡 **Error handling in `create_col`**: Qdrant uses `logger.debug(f"Index for {field} might already exist: {e}")` and swallows index-creation errors. The guide doesn't mention this log-and-swallow pattern for best-effort operations.

---

## Step 5 — Tests (`tests/vector_stores/test_qdrant.py`)

**Guide says:** Use `MagicMock(spec=<ClientClass>)` injected via the `client=` constructor parameter. Test all 11 methods with setup/teardown.

**What Qdrant actually does:**

```python
class TestQdrant(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.qdrant = Qdrant(
            collection_name="test_collection",
            embedding_model_dims=1536,
            client=self.mock_client,   # ← direct injection
        )
```

**Verification:**

| Guide Element | Qdrant Reality | Status |
|---|---|---|
| `MagicMock` for client | ✅ `MagicMock()` used (without `spec=`) | Close match |
| Inject via `client=` param | ✅ `Qdrant(..., client=self.mock_client)` | Match |
| `setUp`/`tearDown` pattern | ✅ `setUp` used; no `tearDown` needed | Match |
| Test all 11 methods | ✅ All methods tested | Match |
| Assert mock calls | ✅ Verifies `.upsert()`, `.search()`, `.delete()`, `.retrieve()`, `.scroll()` etc. | Match |
| `spec=QdrantClient` | ❌ Qdrant tests use plain `MagicMock()` without `spec` | Minor deviation |

**Additional test coverage the guide doesn't mention:**
- 🟡 **Filter tests**: Qdrant tests include `test_search_with_filters` that verifies the `_build_filters` helper constructs the correct Qdrant `Filter` object. The guide doesn't mention testing filter translation.
- 🟡 **`on_disk` tests**: Tests verify `on_disk=True/False` behavior. Guide doesn't mention testing this param.

---

## Summary of All Flags

### 🔴 Steps That Are Wrong or Incomplete

| # | Issue |
|---|---|
| 1 | Guide's `distance` field in config doesn't exist in Qdrant — distance is hardcoded |
| 2 | Guide omits `client` field in config (for pre-built client injection) |
| 3 | Guide omits `on_disk` field in config |
| 4 | Guide omits `validate_extra_fields` validator |
| 5 | Guide uses `mode="after"` validator; Qdrant uses `mode="before"` |
| 6 | Guide's `**kwargs` constructor style is wrong — Qdrant uses explicit params |
| 7 | Guide says use `OutputData` — Qdrant returns native `PointStruct` objects, not `OutputData` |
| 8 | Guide completely omits the filter translation helper (`_build_filters`) |
| 9 | Guide omits payload index creation in `create_col` |

### 🟡 Steps That Are Incomplete (Gaps in Guide)

| # | Issue |
|---|---|
| 10 | Guide doesn't discuss log-and-swallow error handling for best-effort operations |
| 11 | Guide doesn't mention testing filter translation logic |
| 12 | Guide doesn't mention the `on_disk` parameter in tests |
| 13 | Guide says use `spec=ClientClass` in mock — Qdrant uses plain `MagicMock()` |

### ✅ Steps That Are Correct

| # | Element |
|---|---|
| — | File list (5 files) is correct |
| — | `_provider_configs` registration is accurate |
| — | `VectorStoreFactory` registration is accurate |
| — | `try/except ImportError` import guard |
| — | `create_col` called from `__init__` |
| — | All 11 abstract methods are implemented |
| — | `client=` injection used in tests |
| — | `setUp` pattern with `MagicMock` |

## Q15: You just identified specific gaps and inaccuracies in the integration guide by comparing it against the Qdrant implementation. Now revise the guide to incorporate every finding. Correct the inaccuracies, add the missing steps, and remove anything that doesn't reflect how integrations are actually built in this project. Produce the complete revised guide.

# Revised Step-by-Step Integration Guide: Adding a New Vector Store Backend to Mem0

This guide uses **`mystore`** as the example provider name. Every step has been verified against the actual Qdrant implementation — the canonical reference integration in this codebase.

---

## Overview of Required Changes

| Action | File |
|--------|------|
| **CREATE** | `mem0/configs/vector_stores/mystore.py` |
| **CREATE** | `mem0/vector_stores/mystore.py` |
| **CREATE** | `tests/vector_stores/test_mystore.py` |
| **MODIFY** | `mem0/vector_stores/configs.py` |
| **MODIFY** | `mem0/utils/factory.py` |

No other files need to change. The `__init__.py` files in both `mem0/configs/vector_stores/` and `mem0/vector_stores/` are intentionally empty and rely on dynamic imports — do not touch them.

---

## Step 1 — Create the Config Class

**File to create:** `mem0/configs/vector_stores/mystore.py`

This file holds a Pydantic `BaseModel` describing every configuration option for the new backend. Use `mode="before"` validators (not `mode="after"`) so that validation runs on the raw input dict before field assignment, matching how every real config class in the codebase works.

```python
# mem0/configs/vector_stores/mystore.py

from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class MyStoreConfig(BaseModel):
    """Configuration for the MyStore vector store backend."""

    # ── Collection / index settings ──────────────────────────────────────────
    collection_name: str = Field("mem0", description="Name of the collection / index to use")
    embedding_model_dims: Optional[int] = Field(1536, description="Dimensionality of the embedding vectors")

    # ── Optional pre-built client injection ──────────────────────────────────
    # Allows passing an already-constructed client object (e.g., from tests or
    # a shared connection pool) instead of building one from connection params.
    client: Optional[Any] = Field(None, description="Pre-existing MyStore client instance")

    # ── Connection parameters (mutually exclusive groups) ────────────────────
    host: Optional[str] = Field(None, description="MyStore server hostname")
    port: Optional[int] = Field(None, description="MyStore server port")
    url: Optional[str] = Field(None, description="Full URL for MyStore Cloud")
    api_key: Optional[str] = Field(None, description="API key for MyStore Cloud")
    path: Optional[str] = Field(None, description="Local path for file-based / on-disk mode")

    # ── Backend-specific persistence option ──────────────────────────────────
    # Controls whether vectors/payloads are kept on disk vs. in memory.
    # Expose any backend-specific boolean flags here rather than burying them
    # in the implementation.
    on_disk: bool = Field(False, description="Store payloads on disk for durability")

    # NOTE: Do NOT add a `distance` field here. Distance metric is an
    # implementation detail that belongs in the backend class itself
    # (hardcoded or as a local constant). QdrantConfig does not expose it.

    @model_validator(mode="before")
    @classmethod
    def check_connection_params(cls, values: dict) -> dict:
        """Ensure at least one connection method is provided.
        
        VectorStoreConfig will inject path='/tmp/mystore' as a default when no
        other connection params are supplied, so this validator normally only
        fires for explicitly malformed configs.
        """
        has_host = values.get("host") and values.get("port")
        has_url = values.get("url") and values.get("api_key")
        has_path = values.get("path")
        has_client = values.get("client")
        if not any([has_host, has_url, has_path, has_client]):
            raise ValueError(
                "MyStoreConfig requires one of: "
                "(host+port), (url+api_key), path, or a pre-built client."
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: dict) -> dict:
        """Reject any unrecognised configuration keys.
        
        This prevents silent misconfiguration where a typo in a key name causes
        the option to be silently ignored. Every real config class in this
        codebase (QdrantConfig, PineconeConfig, etc.) includes this guard.
        """
        allowed = {
            "collection_name", "embedding_model_dims", "client",
            "host", "port", "url", "api_key", "path", "on_disk",
        }
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"Extra fields not permitted: {extra}")
        return values
```

**Rules:**
- Always include `collection_name` and `embedding_model_dims` — `VectorStoreFactory.create()` calls `config.model_dump()` and passes these by name.
- Always include a `client` field — the mock-injection pattern used throughout the test suite depends on it.
- Always include `on_disk` (or the analogous backend-specific persistence flag) — omitting it forces callers to work around it in the implementation.
- Always include `validate_extra_fields` — the absence of this guard causes silent misconfiguration bugs.
- Use `mode="before"` validators consistently, as all real config classes do.
- Do **not** add a `distance` field — distance metric is hardcoded in the implementation layer.

---

## Step 2 — Register the Config in `VectorStoreConfig`

**File to modify:** `mem0/vector_stores/configs.py`

Add one line to `_provider_configs`. The `model_validator` in `VectorStoreConfig` reads this mapping to dynamically import `mem0.configs.vector_stores.<provider>` and instantiate the correct config class — no further changes are needed in this file.

```python
# mem0/vector_stores/configs.py  — excerpt showing only the diff

_provider_configs: Dict[str, str] = {
    "qdrant": "QdrantConfig",
    "chroma": "ChromaDbConfig",
    # … all existing entries …
    "s3_vectors": "S3VectorsConfig",
    "mystore": "MyStoreConfig",           # ← ADD THIS LINE
}
```

`VectorStoreConfig` also contains an auto-path-injection block that sets `config["path"] = f"/tmp/{provider}"` when `path` is not supplied and the provider config class declares a `path` field. This means users can create a minimal config like `VectorStoreConfig(provider="mystore", config={})` and get a working local backend automatically — but only if you declared `path` as a field in Step 1 (which the template above does).

---

## Step 3 — Register the Implementation in `VectorStoreFactory`

**File to modify:** `mem0/utils/factory.py`

Add one entry to `VectorStoreFactory.provider_to_class`. The value is the fully-qualified dotted path; `load_class()` splits on the last `.` and imports lazily, so the backend library is not imported until this provider is actually instantiated.

```python
# mem0/utils/factory.py  — excerpt showing only the diff

class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "mem0.vector_stores.qdrant.Qdrant",
        "chroma": "mem0.vector_stores.chroma.ChromaDB",
        # … all existing entries …
        "s3_vectors": "mem0.vector_stores.s3_vectors.S3Vectors",
        "mystore": "mem0.vector_stores.mystore.MyStore",   # ← ADD THIS LINE
    }
```

---

## Step 4 — Implement the Backend Class

**File to create:** `mem0/vector_stores/mystore.py`

The class must extend `VectorStoreBase` and implement every abstract method. The structure below follows what Qdrant actually does, including the connection-mode branching, the `is_local` flag, the idempotent collection creation, the filter index creation, and the `shutil.rmtree` clean-slate behaviour for ephemeral local mode.

```python
# mem0/vector_stores/mystore.py

import logging
import shutil
from typing import Optional

try:
    from mystore_client import MyStoreClient  # replace with your actual SDK import
    from mystore_client.models import PointStruct, VectorParams
except ImportError:
    raise ImportError(
        "The `mystore_client` package is required. "
        "Install it with: pip install mystore-client"
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

# Hardcode the distance metric here — do NOT expose it in QdrantConfig.
# Users who need a different metric should create a separate provider entry.
DISTANCE = "cosine"


class MyStore(VectorStoreBase):
    """MyStore vector store backend for Mem0."""

    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        client: Optional[MyStoreClient] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        on_disk: bool = False,
    ):
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims

        # ── Connection mode branching (mirrors Qdrant exactly) ───────────────
        if client is not None:
            # Mode 1: pre-built client injected (used by tests and shared pools)
            self.client = client
            self.is_local = False

        elif host and port:
            # Mode 2: self-hosted server
            self.client = MyStoreClient(host=host, port=port)
            self.is_local = False

        elif url and api_key:
            # Mode 3: managed cloud
            self.client = MyStoreClient(url=url, api_key=api_key)
            self.is_local = False

        else:
            # Mode 4: local file-based storage
            self.is_local = True
            if not on_disk and path:
                # Ephemeral mode: delete any leftover data so we start clean.
                # Qdrant does exactly this with shutil.rmtree(path).
                import os
                if os.path.exists(path):
                    shutil.rmtree(path)
            self.client = MyStoreClient(path=path)

        self.create_col(vector_size=embedding_model_dims, on_disk=on_disk)

    # ── Collection lifecycle ─────────────────────────────────────────────────

    def create_col(self, vector_size: int, on_disk: bool, distance: str = DISTANCE):
        """Create the collection if it does not already exist (idempotent)."""
        existing = self.list_cols()
        for col in existing:
            if col.name == self.collection_name:
                # Already exists — still create filter indexes in case they
                # weren't created on a previous run.
                self._create_filter_indexes()
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
                on_disk=on_disk,
            ),
        )
        self._create_filter_indexes()

    def _create_filter_indexes(self):
        """Create keyword payload indexes on the four standard Mem0 filter fields.

        This is skipped for local instances because the local Qdrant client
        does not support payload indexes. Mirror this guard for your backend.
        """
        if self.is_local:
            return

        # The four metadata fields that mem0/memory/main.py always filters on:
        standard_fields = ["user_id", "agent_id", "run_id", "actor_id"]
        for field in standard_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                # Index may already exist — silently continue.
                pass

    # ── CRUD operations ──────────────────────────────────────────────────────

    def insert(self, vectors, payloads=None, ids=None):
        """Insert or overwrite points using upsert semantics."""
        points = [
            PointStruct(
                id=ids[idx] if ids is not None else idx,
                vector=vector,
                payload=payloads[idx] if payloads else {},
            )
            for idx, vector in enumerate(vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query, vectors, limit=5, filters=None):
        """Return the top-`limit` points nearest to `vectors`.

        Note: `query` is the raw text query string (passed for logging or
        reranking); the actual similarity search uses `vectors` (the embedding).
        """
        query_filter = self._build_filter(filters) if filters else None
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vectors,
            query_filter=query_filter,
            limit=limit,
        )
        return result.points  # return the raw point objects — mem0/memory/main.py
                              # accesses .id, .payload, and .score directly

    def _build_filter(self, filters: dict):
        """Convert a Mem0 filter dict to a backend-native filter object.

        Supports two value shapes:
          - Scalar:  {"user_id": "alice"}              → exact match
          - Range:   {"created_at": {"gte": t0, "lte": t1}} → range condition
        All conditions are ANDed (Filter.must=[...]).
        """
        from mystore_client.models import Filter, FieldCondition, MatchValue, Range
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                conditions.append(
                    FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"]))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions)

    def delete(self, vector_id):
        """Delete a single point by its ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[vector_id],
        )

    def update(self, vector_id, vector=None, payload=None):
        """Update the vector and/or payload for an existing point."""
        if vector is not None:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=vector_id, vector=vector, payload=payload or {})],
            )
        elif payload is not None:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[vector_id],
            )

    def get(self, vector_id):
        """Retrieve a single point by ID."""
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[vector_id],
            with_payload=True,
            with_vectors=True,
        )
        return result[0] if result else None

    def list(self, filters=None, limit=100):
        """List stored points, optionally filtered."""
        query_filter = self._build_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        return list(result[0])

    # ── Collection-level operations ──────────────────────────────────────────

    def list_cols(self):
        """Return a list of collections that exist in the backend."""
        return self.client.get_collections().collections

    def delete_col(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)

    def col_info(self):
        """Return backend-reported metadata about the current collection."""
        return self.client.get_collection(self.collection_name)

    def reset(self):
        """Delete the collection and immediately recreate it (empty)."""
        self.delete_col()
        self.create_col(
            vector_size=self.embedding_model_dims,
            on_disk=False,
        )
```

**Key implementation rules verified against Qdrant:**

| Rule | Why |
|------|-----|
| Support all four connection modes (pre-built client, host+port, url+api_key, path) | `mem0/memory/main.py` may construct the backend in any of these ways |
| Set `self.is_local` based on connection mode | Guards `_create_filter_indexes` — local Qdrant client doesn't support payload indexes |
| Call `self.create_col(...)` at the end of `__init__` | Qdrant does this; the collection must exist before any operation |
| Make `create_col` idempotent | Multiple `Memory` instances (or restarts) hit the same collection |
| Call `_create_filter_indexes()` from `create_col` (not `__init__`) | Covers both the "just created" and "already exists" paths |
| Use `shutil.rmtree` to clear stale local data when `on_disk=False` | Qdrant does exactly this; omitting it causes data contamination across test runs |
| Return raw backend point objects from `search()` | `mem0/memory/main.py` accesses `.id`, `.payload`, `.score` directly (duck-typed) |
| Use `upsert` semantics in `insert()` | Re-inserting the same ID must be safe; Qdrant uses `client.upsert` |
| Hardcode distance metric in the implementation, not in the config | `QdrantConfig` has no `distance` field; it is `Distance.COSINE` in the code |

---

## Step 5 — Write Tests

**File to create:** `tests/vector_stores/test_mystore.py`

Inject a `MagicMock(spec=MyStoreClient)` via the `client=` parameter — this is the canonical Mem0 test pattern (see `tests/vector_stores/test_qdrant.py`). Using `spec=MyStoreClient` restricts the mock to the real client's API, catching interface drift early.

```python
# tests/vector_stores/test_mystore.py

import unittest
from unittest.mock import MagicMock, call

from mystore_client import MyStoreClient

from mem0.vector_stores.mystore import MyStore


class TestMyStore(unittest.TestCase):

    def setUp(self):
        # Inject a spec-constrained mock so attribute typos raise AttributeError.
        self.mock_client = MagicMock(spec=MyStoreClient)

        # Stub out list_cols so create_col sees an empty collection list,
        # triggering the creation branch during __init__.
        self.mock_client.get_collections.return_value.collections = []

        self.store = MyStore(
            collection_name="test_collection",
            embedding_model_dims=128,
            client=self.mock_client,   # ← pre-built client injection
        )

    # ── Initialisation ───────────────────────────────────────────────────────

    def test_create_col_called_on_init(self):
        """__init__ must call create_collection exactly once."""
        self.mock_client.create_collection.assert_called_once()

    def test_no_filter_indexes_when_is_local(self):
        """_create_filter_indexes must be a no-op for local instances.
        
        The is_local path is set when a path is given with no client/host/url.
        We test it separately here by patching shutil.rmtree.
        """
        from unittest.mock import patch
        self.mock_client.get_collections.return_value.collections = []
        with patch("shutil.rmtree"), patch("os.path.exists", return_value=True):
            local_store = MyStore(
                collection_name="local_test",
                embedding_model_dims=128,
                client=None,       # force local-path branch
                path="/tmp/test_mystore",
            )
        # For a local store, create_payload_index should never be called.
        local_store.client.create_payload_index = MagicMock()
        local_store._create_filter_indexes()
        local_store.client.create_payload_index.assert_not_called()

    # ── insert ───────────────────────────────────────────────────────────────

    def test_insert_calls_upsert_with_correct_points(self):
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"user_id": "alice"}, {"user_id": "bob"}]
        ids = ["id-1", "id-2"]

        self.store.insert(vectors=vectors, payloads=payloads, ids=ids)
        self.mock_client.upsert.assert_called_once()
        args, kwargs = self.mock_client.upsert.call_args
        self.assertEqual(kwargs.get("collection_name", args[0] if args else None),
                         "test_collection")

    def test_insert_without_ids_uses_index(self):
        """When no ids are given the loop index is used as point ID."""
        self.store.insert(vectors=[[0.1, 0.2]])
        self.mock_client.upsert.assert_called_once()

    # ── search ───────────────────────────────────────────────────────────────

    def test_search_no_filter(self):
        mock_points = [MagicMock(id="id-1", payload={"memory": "likes cats"}, score=0.9)]
        self.mock_client.query_points.return_value.points = mock_points

        results = self.store.search(query="cats", vectors=[0.1, 0.2], limit=5)

        self.mock_client.query_points.assert_called_once_with(
            collection_name="test_collection",
            query=[0.1, 0.2],
            query_filter=None,
            limit=5,
        )
        self.assertEqual(results, mock_points)

    def test_search_with_exact_filter(self):
        self.mock_client.query_points.return_value.points = []
        self.store.search(
            query="q", vectors=[0.0], limit=3,
            filters={"user_id": "alice"},
        )
        _, kwargs = self.mock_client.query_points.call_args
        self.assertIsNotNone(kwargs["query_filter"])

    def test_search_with_range_filter(self):
        self.mock_client.query_points.return_value.points = []
        self.store.search(
            query="q", vectors=[0.0], limit=3,
            filters={"created_at": {"gte": 100, "lte": 200}},
        )
        _, kwargs = self.mock_client.query_points.call_args
        self.assertIsNotNone(kwargs["query_filter"])

    # ── delete / update / get ─────────────────────────────────────────────────

    def test_delete(self):
        self.store.delete("id-1")
        self.mock_client.delete.assert_called_once_with(
            collection_name="test_collection",
            points_selector=["id-1"],
        )

    def test_update_vector_and_payload(self):
        self.store.update("id-1", vector=[0.5, 0.6], payload={"x": 1})
        self.mock_client.upsert.assert_called()

    def test_get(self):
        mock_point = MagicMock()
        self.mock_client.retrieve.return_value = [mock_point]
        result = self.store.get("id-1")
        self.assertEqual(result, mock_point)

    def test_get_missing_id_returns_none(self):
        self.mock_client.retrieve.return_value = []
        result = self.store.get("missing")
        self.assertIsNone(result)

    # ── collection operations ─────────────────────────────────────────────────

    def test_list_cols(self):
        mock_cols = [MagicMock(name="test_collection")]
        self.mock_client.get_collections.return_value.collections = mock_cols
        result = self.store.list_cols()
        self.assertEqual(result, mock_cols)

    def test_delete_col(self):
        self.store.delete_col()
        self.mock_client.delete_collection.assert_called_once_with("test_collection")

    def test_col_info(self):
        self.store.col_info()
        self.mock_client.get_collection.assert_called_once_with("test_collection")

    def test_reset_recreates_collection(self):
        self.mock_client.get_collections.return_value.collections = []
        self.store.reset()
        self.mock_client.delete_collection.assert_called()
        # create_collection called once at init + once after reset
        self.assertEqual(self.mock_client.create_collection.call_count, 2)


if __name__ == "__main__":
    unittest.main()
```

---

## Summary of Changes from the Original Guide

The following inaccuracies and omissions have been corrected:

| Issue | Original Guide | This Revised Guide |
|-------|---------------|--------------------|
| `distance` field in config | Incorrectly included in `MyStoreConfig` | Removed — it is hardcoded in the implementation (`Distance.COSINE` in Qdrant); not a config-layer concern |
| `client` field in config | Missing | Added — required for the mock-injection test pattern used across the entire test suite |
| `on_disk` flag in config | Missing | Added — Qdrant exposes this; omitting it forces backend-specific options into the implementation with no user control |
| `validate_extra_fields` validator | Missing entirely | Added — every real config class in the codebase includes this guard to prevent silent misconfiguration |
| Validator `mode` | Single `mode="after"` validator | Two `mode="before"` validators — matching how all real config classes are written |
| `is_local` flag in implementation | Not mentioned | Added — guards `_create_filter_indexes` (local clients don't support payload indexes) |
| `_create_filter_indexes` | Not mentioned | Added — required for performant filtered queries on remote instances; called from `create_col`, not `__init__` |
| `shutil.rmtree` clean-slate behaviour | Not mentioned | Added — Qdrant deletes stale local data when `on_disk=False`; omitting this causes data contamination across runs |
| Return type of `search()` | Implied `OutputData` normalization | Corrected — return raw backend point objects (duck-typed); `mem0/memory/main.py` accesses `.id`, `.payload`, `.score` directly |
| Test mock pattern | Generic — not spec-constrained | `MagicMock(spec=MyStoreClient)` — matching `test_qdrant.py`; catches interface drift at test time |
| Filter construction | Not shown | Added `_build_filter` supporting both exact-match and range conditions, matching Qdrant's `_create_filter` |
