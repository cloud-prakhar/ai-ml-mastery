# Diagram Template & Standards

How diagrams are built in this repository, so that every one of them looks like it belongs to the
same book. **Mermaid is the default.** Do not use ASCII art where Mermaid explains it better.

---

## The four deliverables per major concept

For every major concept, produce:

1. **Technical Mermaid diagram** — the accurate structure
2. **Simple analogy diagram** — the same shape, everyday nouns
3. **Infographic-generation prompt** — stored in [`40-visual-learning/image-prompts/`](../40-visual-learning/image-prompts/)
4. **9:16 learning-card layout** — a vertical revision card suggestion

---

## Hard rules

- ✅ Must render on GitHub. Preview it. Do not assume.
- ✅ Short, readable labels.
- ✅ Consistent styling (palette below).
- ❌ No unsupported Mermaid syntax.
- ❌ **No unescaped `(` `)` `&` `"` `:` inside node labels** — the most common cause of a broken
  diagram. Write `M15[15 Embeddings and Vector Search]`, not `M15[15 Embeddings & Vector Search (ANN)]`.
  If you need punctuation, wrap the label in double quotes: `A["Query (user input)"]`.
- ❌ No diagram with more than ~15 nodes. Split it instead.
- ❌ No text so small or dense that it needs zooming.

---

## Standard palette

Consistency matters more than beauty. Use these five roles:

| Role | Fill | Stroke | Text | Used for |
| --- | --- | --- | --- | --- |
| Input / source | `#dbeafe` | `#2563eb` | `#1e3a8a` | user input, raw data |
| Processing | `#fef3c7` | `#d97706` | `#78350f` | transformation, computation |
| Model | `#fae8ff` | `#a21caf` | `#701a75` | any model or inference step |
| Storage | `#d1fae5` | `#059669` | `#064e3b` | databases, indexes, caches |
| Output / risk | `#fee2e2` | `#dc2626` | `#7f1d1d` | results, errors, danger points |

```mermaid
flowchart LR
    A[User input]:::input --> B[Preprocess]:::process
    B --> C[Model]:::model
    C --> D[(Vector store)]:::storage
    D --> E[Response]:::output

    classDef input fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef process fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef model fill:#fae8ff,stroke:#a21caf,color:#701a75
    classDef storage fill:#d1fae5,stroke:#059669,color:#064e3b
    classDef output fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
```

---

## Diagram type by purpose

| You want to show… | Use |
| --- | --- |
| A pipeline or process | `flowchart LR` |
| A hierarchy or dependency | `flowchart TD` |
| Interaction over time between components | `sequenceDiagram` |
| Lifecycle with modes | `stateDiagram-v2` |
| A concept's branches | `mindmap` |
| Two options compared | side-by-side `subgraph` blocks |
| A schedule or phased plan | `gantt` |
| Data relationships | `erDiagram` |

---

## Reference patterns

### Training pipeline

```mermaid
flowchart LR
    A[Raw data]:::input --> B[Clean and validate]:::process
    B --> C[Split train/val/test]:::process
    C --> D[Feature engineering]:::process
    D --> E[Train]:::model
    E --> F{Meets threshold?}
    F -->|No| G[Tune hyperparameters]:::process
    G --> E
    F -->|Yes| H[(Model registry)]:::storage

    classDef input fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef process fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef model fill:#fae8ff,stroke:#a21caf,color:#701a75
    classDef storage fill:#d1fae5,stroke:#059669,color:#064e3b
```

### Inference pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant C as Cache
    participant M as Model
    U->>A: Request
    A->>A: Validate input
    A->>C: Check cache
    alt Cache hit
        C-->>A: Cached result
    else Cache miss
        A->>M: Inference
        M-->>A: Prediction
        A->>C: Store result
    end
    A-->>U: Response
```

### Comparison

```mermaid
flowchart TD
    subgraph RAG["Retrieval-Augmented Generation"]
        R1[Add knowledge at query time]
        R2[Updates instantly]
        R3[Higher per-query latency]
    end
    subgraph FT["Fine-Tuning"]
        F1[Bake behaviour into weights]
        F2[Updates require retraining]
        F3[Lower per-query latency]
    end
```

### Analogy diagram (paired with a technical one)

Technical:
```mermaid
flowchart LR
    Q[Query] --> E[Embed] --> S[(Vector index)] --> R[Top-k chunks] --> L[LLM] --> A[Answer]
```

Analogy:
```mermaid
flowchart LR
    Q2[Your question] --> E2[Librarian understands it] --> S2[(Library shelves)] --> R2[Few relevant books] --> L2[Expert reads them] --> A2[Answer with sources]
```

---

## Image-prompt template

Save as `40-visual-learning/image-prompts/{concept}.md`:

```markdown
# Image prompt — {Concept}

**Purpose:** {what the picture must teach}
**Audience:** 🟢 beginner
**Aspect ratio:** 16:9 (documentation) / 9:16 (learning card)

## Prompt

A friendly, modern, flat-illustration diagram explaining {concept}.
Style: clean vector, high contrast, generous whitespace, rounded shapes,
soft shadows, educational infographic aesthetic, minimal text.
Palette: blue #2563eb, amber #d97706, purple #a21caf, green #059669 on a light background.
Show: {element 1} on the left flowing to {element 2} in the centre and {element 3} on the right,
connected by clear arrows. Label each element with two or three words maximum.
No photorealism. No clutter. No small print. No fake logos or brand marks.

## Text overlay (keep it this short)
- Title: {4 words max}
- Labels: {2-3 words each}

## Accuracy checks before use
- [ ] Arrows point the correct way
- [ ] No invented or misleading component appears
- [ ] Any text in the image is spelled correctly
- [ ] Nothing implies endorsement by a real company
```

## 9:16 learning-card layout

```
┌─────────────────┐
│  🧠 CONCEPT     │  ← title, 4 words max
├─────────────────┤
│                 │
│    [diagram]    │  ← the analogy version, not the technical one
│                 │
├─────────────────┤
│ 🍰 In one line: │
│ {plain-English  │
│  sentence}      │
├─────────────────┤
│ ⚠️ Gotcha:      │
│ {one mistake}   │
├─────────────────┤
│ ai-ml-mastery   │
└─────────────────┘
```

---

## Pre-commit checklist

- [ ] Renders on GitHub (previewed, not assumed)
- [ ] No unescaped special characters in labels
- [ ] Uses the standard palette
- [ ] Fewer than ~15 nodes
- [ ] Arrows point the right way
- [ ] Labels are short and correct
- [ ] Major concepts also have an analogy diagram and an image prompt
- [ ] Source `.mmd` saved to [`diagrams/`](../diagrams/) if reused across files

---

[🏠 Repository Home](../README.md) · [Visual Learning module →](../40-visual-learning/README.md)
