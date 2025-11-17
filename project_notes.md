

## ✅ **Core Project Goal & Contrast**

The project asks you to **explain a generative model** (LLM) using **Point-wise Mutual Information (PMI)** between **masked words** in a sentence. This method explains **dependencies between generated words**, which is different from LIME/SHAP, as we are **not** explaining *predictions* (like a classification score).

---

## 1. **Point-wise Mutual Information (PMI)** — The Core Concept

PMI quantifies the statistical dependence between two words, $w_i$ and $w_j$. A high PMI value indicates a strong, positive relationship between their occurrences.

### 💡 Key Formula & Implementation

For two words ($w_i$) and ($w_j$):
$$
PMI(w_i, w_j) = \log \frac{P(w_i, w_j)}{P(w_i) P(w_j)}
$$

* **Process:** You must **mask two words** (anchor word and target word), ask the LLM to fill them in **many times** (e.g., 10 / 20 samples), and use the counts from these samples to approximate the required probabilities.

### 🧮 Practical Computation from Samples

Given $N$ total samples, and using $C$ for counts:

* $C_{ij}$ = number of times LLM filled mask pair with ($w_i, w_j$).
* $C_i$ = number of times LLM filled mask-1 with word $w_i$.
* $C_j$ = number of times LLM filled mask-2 with word $w_j$.

The probabilities are approximated as:
$$
P(w_i, w_j) = \frac{C_{ij}}{N} \quad P(w_i) = \frac{C_i}{N} \quad P(w_j) = \frac{C_j}{N}
$$

The final PMI formula you should **implement exactly** is:
$$
PMI(w_i,w_j) = \log \frac{C_{ij} \cdot N}{C_i \cdot C_j}
$$

---

## 2. **What Your Code Must Compute (Exercise 3.4)**

Your main task is to process the raw LLM responses and calculate the normalized PMI values for visualization.

Assume you have the raw outputs: `all_responses = [[(w1, w2), (w1, w2)...], ...]` (where each sublist is for a fixed anchor word).

### ⚙️ Steps for Computation

1.  **Step 1 — Build Frequency Tables:**
    * For a given target word position $j$, extract all LLM outputs for mask-1 and mask-2.
    * Build the necessary count dictionaries for the PMI formula: `count_w1[word]`, `count_w2[word]`, and `count_pair[(word1, word2)]`.

2.  **Step 2 — Compute PMI:**
    * For each word ($w_j$) in the sentence, compute $PMI(\text{anchor\_word}, w_j)$ using the counts from Step 1.

3.  **Step 3 — Normalize:**
    * Since PMI ranges from $(-\infty, +\infty)$, you must normalize the values for visualization.
    * **Common choice (maps values to [0, 1]):**
        ```python
        normalized = (PMI - min_PMI) / (max_PMI - min_PMI)
        ```

4.  **Step 4 — Visualize:**
    * Use the normalized values to color-code the sentence.
    * **Visualization Key:** **Light color** = low PMI (weak dependency), **Dark color** = high PMI (strong dependency).

---

## 3. **Context & Tools**

### 🧩 Tokenization
* The notebook performs **very basic preprocessing** (lowercasing, splitting on whitespace).
* You do **NOT** need complex transformers tokenization for the core exercises.

### 🤖 Model Usage
* The model interaction is: `response = model.get_response(prefix_prompt(prompt))`.
* The `prompt` contains `[MASK-1]` and `[MASK-2]`.
* `prefix_prompt` adds instructions (e.g., "Fill in the masks only").
* `get_replacements()` extracts the two filled words from the raw `response`.

### 🧪 Experiment Design
* For a sentence of length $L$:
    * Fix one word as the **anchor index $i$**.
    * For **every other index $j$**, generate a masked sentence with masks at $i$ and $j$.
    * Collect 20 responses per pair.
    * Compute $PMI(\text{anchor\_word}, \text{word}_j)$.

---

## 4. **Interpretation & Limitations (Exercise 4)**

### 🧠 How to Interpret the Output
* **High PMI** means: The LLM thinks the masked word **strongly depends** on the anchor word.
* **Example Interpretation:**
    * Sentence: *“Tokyo is the capital city of Japan.”*
    * Anchor = “Tokyo”
    * **Expected High PMI** with: *Japan, capital, city*.
    * **Expected Low PMI** with: *is, the, of*.

### ⚠️ Key Limitations You Must Understand
* **Sample Size:** PMI is unreliable if the count $N$ (e.g., 20) is too small.
* **Sparsity:** PMI is unreliable if counts ($C_i, C_j, C_{ij}$) are small (sparse data).
* **Semantics:** LLM might generate synonyms (e.g., *metropolis* vs *city*). Basic string matching will miss these.
* **Grammar:** Masking breaks the sentence's grammar, which can distort LLM behavior.
* **Noise:** Generative models do not always fill both masks, and the replacement extraction can be noisy.

---

## 5. **Bonus Exercises (Further Understanding)**

### 5.1 Tokenization (Beyond Basic)
* Need to handle: **contractions** (“I’ll” $\rightarrow$ “I” + “will”), **multi-word expressions** (“New York” $\rightarrow$ single unit), and **morphological normalization** (stemming/lemmatization).
* Tools: spaCy, NLTK, TextBlob.

### 5.2 Semantic Matching (Beyond Exact String Match)
* Instead of exact string comparison, use **word embeddings** (Word2Vec, GloVe).
* Compute a **soft match score** using the cosine similarity:
$$
\text{similarity}(w_1, w_2) = \frac{\vec{w_1} \cdot \vec{w_2}}{| \vec{w_1}|| \vec{w_2}|} \in [0,1]
$$

---

Do you want the full implementation code for **Exercise 3.4** next?