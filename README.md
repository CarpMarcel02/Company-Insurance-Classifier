# Company Insurance Taxonomy Classifier

This repository contains my solution for building a robust company classifier for a static insurance taxonomy.

Given a list of companies (description, business tags, sector/category/niche) and a list of taxonomy labels, the goal is to assign one or more relevant insurance labels per company. Since no ground-truth mapping is provided, the solution focuses on explainability, qualitative validation, and failure-mode analysis.

---

## Files
- `Company-Insurance-Classifier.ipynb` – end-to-end notebook (data cleaning, modeling, evaluation, final export)
- `companies_with_labels.csv` – final annotated output (includes `insurance_labels`)

---

## Approach

### 1) Data preprocessing
- Removed rows with missing critical fields (`description`, `business_tags`, `sector`, `category`, `niche`)
- Removed rows with empty business tags (`[]`)
- Removed fully duplicated rows
- Normalized `business_tags` into clean text
- Built a single text feature `full_text` by concatenating:
  `description + business_tags + sector + category + niche`

### 2) Baseline #1: TF–IDF + cosine similarity (lexical matching)
- Vectorized company texts and label texts using TF–IDF (unigrams + bigrams)
- Computed cosine similarity between each company and each label
- Selected up to top-3 labels per company (with similarity scores)

Strength: performs best when there is strong keyword overlap between the company text and label names.

### 3) Baseline #2: Semantic embeddings + cosine similarity
- Used a pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`) to generate semantic embeddings
- Computed cosine similarity and selected up to top-3 labels

Strength: can help when the wording differs (synonyms / different phrasing) and keyword overlap is weak.

### 4) Final model: Hybrid strategy + `Unknown`
To combine both strengths, I use a simple fallback rule:
- If TF–IDF top-1 similarity ≥ `0.10`, use TF–IDF prediction
- Otherwise, if embeddings top-1 similarity ≥ `0.25`, use embedding prediction
- Otherwise, output `Unknown`

Final predictions are stored in `insurance_labels_final`.

---

## Evaluation / Validation (no ground truth)
Because the task has no predefined ground truth, I validate the solution through:
- Manual inspection of random samples and low-score examples
- Similarity score distribution plots (confidence proxy)
- Worst-case analysis (lowest TF–IDF matches) to see whether embeddings help
- Inspection of `Unknown` cases to confirm taxonomy-mismatch / low-confidence scenarios

Key limitation: if a business type is not covered by the taxonomy, both methods may produce weak or misleading matches. In those cases, `Unknown` is preferable to forcing an incorrect label.

---

## How to run
Open `Company-Insurance-Classifier.ipynb` in Google Colab and run cells top-to-bottom.  
The notebook exports the final annotated dataset as `companies_with_labels.csv`.
