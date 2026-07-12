# Schema Vocabulary Handoff — for `SchemaLLMPathExtractor`

> Purpose: specify the **exact structure** the GraphRAG pipeline needs for its extraction schema, so the vocabulary can be regenerated in the right shape. The two YAKE keyword files delivered so far (`rag_vocabulary.json`, `test_vocabulary.json`) are the wrong shape — this document explains why and what to produce instead. It also includes a **complete proposed schema** that can be used directly or refined.

---

## 1. Context

We are building a knowledge graph over a technical-book library using **LlamaIndex `SchemaLLMPathExtractor` → Neo4j**. That extractor is configured with a **constrained schema**: two small fixed lists — **entity types** and **relation types** — that are enforced via the LLM's structured output. This is what keeps the graph traversable. Free-form extraction produced ~2 relations per relation-type (437 distinct types over 855 relations on 100 chunks); a constrained schema yields dozens of edges per type.

In code (`ingest.py`) the schema is literally two lists:

```python
SCHEMA_ENTITIES  = ["Concept", "Technique", ...]     # entity TYPES
SCHEMA_RELATIONS = ["USES", "IMPLEMENTS", ...]        # relation TYPES
```

---

## 2. What was delivered vs. what's needed

| | Delivered (YAKE) | Needed |
|---|---|---|
| Content | 833 / 67 specific **keywords** ("binary search tree", "factory method pattern") | ~10–20 entity **TYPES** + ~10–20 relation **TYPES** |
| Entity level | Instances / names | Categories the extractor sorts names into |
| Relations | **None** | **Required** — this is the whole point |
| Noise | OCR errors, colophon cities ("beijing boston farnham"), watermark text ("aaron bagay"), example proper nouns ("ravena coeymans selkirk") | Curated, reusable types |

Two critical gaps:

1. **Types, not instances.** `SchemaLLMPathExtractor` does *not* pick entities from a fixed list. It **generates** entity names (e.g. "binary search tree") and assigns each a **type** from a small set (e.g. `DataStructure`). The YAKE lists are names; we need the *types*.
2. **No relations.** The noise problem was entirely relation-type sprawl. A vocabulary with entities but no relations does not address it. **Relation types are the most important deliverable.**

Relations can't come from keyword frequency — they require reading *how ideas connect* ("a hash table `IMPLEMENTS` a map", "microservices `ALTERNATIVE_TO` a monolith"). That's the work still to do.

---

## 3. Required output structure

One JSON file per schema (a **test** schema and a **target** schema), each in this shape:

```json
{
  "entity_types": [
    { "type": "DataStructure",
      "definition": "A concrete way of organizing data (lists, trees, tables, heaps).",
      "examples": ["binary search tree", "hash table", "linked list"] }
  ],
  "relation_types": [
    { "type": "IMPLEMENTS",
      "definition": "Subject provides a concrete realization of the object.",
      "example": "DataStructure IMPLEMENTS Concept  (a hash table implements a map)" }
  ]
}
```

**Rules**
- **Entity types:** PascalCase (`DataStructure`), ~10–20 total. Each should cover **many** instances (dozens+ across the corpus). If a type would match only 1–2 things, it's too fine — merge it up.
- **Relation types:** UPPER_SNAKE (`DEPENDS_ON`), ~10–20 total, **directional** (subject → object), reusable across books. Always include a `RELATED_TO` catch-all.
- The `examples` / `example` fields are guidance for the LLM prompt — the YAKE keywords are perfect raw material for these.

---

## 4. How to derive them (method)

1. **Cluster the existing YAKE keywords into types.** The 833 terms already reveal the types by grouping: BST/linked-list/hash-table → `DataStructure`; factory/visitor/strategy → `DesignPattern`; microservices/layered/event-driven → `ArchitectureStyle`; SQL/MySQL/Hadoop/Spark → `Technology`; etc.
2. **Read a sample of each book for the *relations*.** Skim a chapter per book and note the verbs connecting concepts (defines, implements, depends on, contrasts with, is an example of…). Collapse synonyms into ~15 relation types.
3. **Keep it small and reusable.** The goal is a *bounded* vocabulary. Prefer general types that recur over specific ones that appear once.
4. **Filter noise.** Drop OCR errors, publisher colophon text, license/watermark strings (including personal names), and proper nouns from worked examples.

---

## 5. Complete proposed schema (use directly or refine)

### 5a. TEST schema — *A First Course in Linear Algebra* + *Python Programming With Design Patterns* (math + software)

```json
{
  "entity_types": [
    {"type": "Concept",     "definition": "A general idea or object of study.", "examples": ["vector space", "linear transformation", "polymorphism"]},
    {"type": "Theorem",     "definition": "A proven mathematical statement.", "examples": ["rank-nullity theorem"]},
    {"type": "Definition",  "definition": "A formal definition of a term.", "examples": ["invertible linear transformation"]},
    {"type": "Method",      "definition": "A procedure or technique.", "examples": ["matrix multiplication", "Gaussian elimination"]},
    {"type": "Notation",    "definition": "A symbol or notational convention.", "examples": ["archetype label"]},
    {"type": "DesignPattern","definition": "A reusable software design pattern.", "examples": ["factory method", "visitor", "singleton"]},
    {"type": "Class",       "definition": "A code class/object in an example.", "examples": ["Employee class", "command object"]},
    {"type": "Person",      "definition": "An author or named figure.", "examples": ["Rob Beezer"]},
    {"type": "Work",        "definition": "A referenced book/paper.", "examples": ["A First Course in Linear Algebra"]},
    {"type": "Example",     "definition": "A worked example or archetype.", "examples": ["linear algebra archetype"]}
  ],
  "relation_types": [
    {"type": "DEFINES",     "definition": "Subject gives the meaning of the object.", "example": "Definition DEFINES Concept"},
    {"type": "PROVES",      "definition": "Subject establishes the object.", "example": "Theorem PROVES Concept"},
    {"type": "DEPENDS_ON",  "definition": "Subject requires the object.", "example": "Theorem DEPENDS_ON Definition"},
    {"type": "IS_A",        "definition": "Subject is a kind of object.", "example": "Concept IS_A Concept"},
    {"type": "PART_OF",     "definition": "Subject is a component of object.", "example": "Method PART_OF Concept"},
    {"type": "EXAMPLE_OF",  "definition": "Subject illustrates object.", "example": "Example EXAMPLE_OF Concept"},
    {"type": "IMPLEMENTS",  "definition": "Subject realizes object in code.", "example": "Class IMPLEMENTS DesignPattern"},
    {"type": "USES",        "definition": "Subject employs object.", "example": "DesignPattern USES Class"},
    {"type": "GENERALIZES", "definition": "Subject is a broader form of object.", "example": "Concept GENERALIZES Concept"},
    {"type": "AUTHORED_BY", "definition": "Work written by Person.", "example": "Work AUTHORED_BY Person"},
    {"type": "RELATED_TO",  "definition": "Catch-all association.", "example": "Concept RELATED_TO Concept"}
  ]
}
```

### 5b. TARGET schema — the 61-book library (Python, DSA, data science, software eng, architecture, databases)

**Entity types (14):** `Concept, Algorithm, DataStructure, DesignPattern, ArchitectureStyle, Technology, Principle, Technique, Metric, QualityAttribute, Person, Work, Field, Visualization`

**Relation types (15):** `IS_A, PART_OF, USES, IMPLEMENTS, SOLVES, IMPROVES, ALTERNATIVE_TO, CONTRASTS_WITH, DEPENDS_ON, EXAMPLE_OF, DEFINES, MEASURED_BY, APPLIES_TO, AUTHORED_BY, RELATED_TO`

> `ALTERNATIVE_TO` / `CONTRASTS_WITH` are deliberately included — the architecture books (*The Hard Parts*, *Fundamentals of Software Architecture*) are largely about tradeoffs, and those relations capture the most valuable structure in that material. `Visualization` covers the data-viz-heavy Phase 6.

(Full definitions/examples for the target set to be filled in from the YAKE keyword clusters — the keywords in `rag_vocabulary.json` map cleanly onto these 14 types.)

---

## 6. Deliverable

Two files, each in the §3 structure:
- `test_schema.json` — for the 2-book validation set (§5a is ready to use).
- `target_schema.json` — for the 61-book library (§5b types, with definitions/examples fleshed out).

The pipeline consumes these as `SCHEMA_ENTITIES` (the `type` values) and `SCHEMA_RELATIONS` (the `type` values); the definitions/examples become prompt guidance.
