# Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?

Reading Comprehension (RC) item difficulty is closely linked to the **cognitive complexity** required to process and verify information in the passage.  
This repository evaluates whether **Large Language Models (LLMs)** can reliably estimate such complexity through two key dimensions: **Evidence Scope** and **Transformation Level**.

---

## 1. Background

### Evidence Scope
Items that can be solved by referencing a **single sentence** are generally easier than those requiring the **integration of information scattered across multiple sentences**.  
In this dataset, *Evidence Scope* refers to the **span of text necessary to determine the truth value of a statement**, and it is categorized into three levels:

| Level | Description |
|:--|:--|
| **Single-sentence evidence** | All information needed to evaluate the statement is contained within a single sentence in the passage. |
| **Multi-sentence evidence** | The information is distributed across multiple sentences, requiring inter-sentential integration. |
| **Insufficient evidence** | The passage lacks enough information to confirm or reject the statement. Learners must examine the entire passage before concluding that no supporting evidence exists. |

### Transformation Level
When the degree of **transformation** between a statement and its supporting evidence increases, verifying the statement’s truth value becomes more cognitively demanding.  
We adopt a **five-level taxonomy** capturing the type and extent of transformation needed to derive the statement from its evidence:

| Level | Description |
|:--|:--|
| **Word Matching** | Content words in the statement appear verbatim and in the same order as in the passage; no transformation is required. |
| **Transformed Word Matching** | The same content words are present but have been reordered. |
| **Paraphrasing** | The statement uses semantically equivalent expressions while maintaining the phrase order. |
| **Transformed Paraphrasing** | Both paraphrasing and reordering occur, involving lexical and structural transformation. |
| **Inference** | The statement cannot be derived directly from surface forms in the passage—even through paraphrasing or reordering—and requires inference. |

These two dimensions together capture how much **information integration** and **semantic transformation** an RC item demands, thereby serving as cognitive complexity indicators relevant to prior difficulty estimation.

---

## 2. Dataset

**Source**: The dataset was reconstructed from a subset of reading comprehension (RC) items in **RACE**.

**Usage restriction**: In accordance with the original license, the dataset is available **for non-commercial research use only**. Redistribution or commercial use of any portion of the data is prohibited.

**Files:**
- `data/ReCo.demo.json` — small demo set for quick execution  
- `data/ReCo.test.json` — main evaluation/reporting set  

Each JSON file includes entries with the following fields (example below).

```json
{
    "id": "d1|true|middle6854.txt",
    "question": "Which of the following is TRUE?",
    "passage": [
        "It is very important for children to get to school safely and on time every day.",
        "Luckily, there is a new program called Free Home to School Transport .",
        "It gives children free rides to school.",
        ...
    ],
    "answer_position": "B",
    "position": "C",
    "factuality": "Not True",
    "statement": "Poor students can not have free transport to school.",
    "true_version": "Poor students can have free transport to school.",
    "reasoning_complexity": "single_Transformed Paraphrasing",
    "evidence": [
        "Also, there are still free home to school transport for children in poor families and children with special educational needs, you can find out more on the Internet and see if your children are qualified."
    ]
}
```

---

## 4. Quick Start

Run both dimensions (Evidence Scope and Transformation Level):

```bash
sh run.sh qwen-m 1 es.sp few
```

Run specific sub-task:

```bash
sh run_fine.sh qwen-m 1 falsify few
```
