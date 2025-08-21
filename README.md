# Eagle Router

A model routing system that uses embedding retrieval to consider local ELO score of the given prompt along with global ELO score. [Paper](https://arxiv.org/abs/2409.15518)

## 📋 Prerequisites

- Python 3.12+
- MongoDB Atlas account with a cluster configured for vector search
- Required Python packages (install via `uv` or `pip`)

## 🛠️ Installation

1. Install dependencies:
```bash
uv pip install -e .
# or
pip install -r requirements.txt
```

2. Set up MongoDB Atlas:
   - Create a MongoDB Atlas cluster
   - Enable Vector Search on your cluster
   - Create a vector search index on the `training_samples` collection with:
     - Field: `embedding`
     - Similarity: `euclidean`
     - Index name: `vector_index`

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your MongoDB connection string
```

## 🎯 Usage

### Basic Example

```python
from eagle_router import EagleRouter
import numpy as np

# Initialize router
router = EagleRouter(
    mongodb_uri="your-mongodb-uri",
    database_name="eagle_router",
    P=0.5,  # Weight for global vs local scores
    N=20,   # Number of nearest neighbors
    K=32    # ELO sensitivity
)

# Register models
router.add_model("gpt-4o-mini")
router.add_model("claude-3-haiku")

# Add training samples
embedding = np.random.randn(1536).astype(np.float32)
router.add_training_sample(
    prompt="Write a Python function",
    embedding=embedding,
    model_a="gpt-4o-mini",
    model_b="claude-3-haiku",
    score=0  # 0=model_a wins, 1=model_b wins, 2=draw
)

# Route a query
query_embedding = np.random.randn(1536).astype(np.float32)
scores = router.route(query_embedding)
print("Scores:", scores)
best_model = router.get_best_model(query_embedding)
print(f"Best model: {best_model}")
```

### Scoring Algorithm

The router combines:
- **Global Scores**: ELO ratings calculated from all training data
- **Local Scores**: ELO ratings calculated from K nearest neighbors with embedding vector of provided prompt against previous match results' prompts.
- **Final Score**: `P * global_score + (1-P) * local_score`

### Indirect Scoring

When models haven't been directly compared, the system uses graph traversal to find monotonic paths through the ELO matrix:
- Non-increasing paths → use minimum ELO
- Non-decreasing paths → use maximum ELO


## 🎛️ Configuration Parameters

- **P** (0-1): Weight for global vs local scores. Higher = more weight on global performance
- **N**: Number of nearest neighbors for local scoring
- **K**: ELO sensitivity (typically 16-32). Higher = more volatile ratings