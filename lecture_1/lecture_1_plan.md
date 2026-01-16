# Lecture 1: Introduction to ML for Language

## Course Information

**Duration:** 60-75 minutes
**Prerequisites:** Basic Python programming
**Next Lecture:** Neural Networks for NLP

---

## Lecture Outline

### 1. Motivation: Why ML for Language? (5 min)

**Topics:**

- **Success stories:** ChatGPT, translation, sentiment analysis
- **This lecture:** Build two models from scratch
  - Text classifier (sentiment/spam detection)
  - Language model (predicting next words)
- Course roadmap: From simple models → transformers → LLMs

---

### 2. Essential Libraries (10 min)

**Topics:**

- **NumPy:** Arrays and numerical operations
- **scikit-learn:** ML models and datasets
- **Collections:** Counting and dictionaries
- **Code examples:** 2-3 line cells showing each library

**Demonstrations:**

- Load text data
- Basic array operations
- Simple text processing

---

### 3. Problem Setup: What is ML? (5 min)

**Topics:**

- **Input → Output:** Text → Label
- **Example:** "I love this movie!" → positive sentiment
- **Learning:** Find patterns in training data
- **Dataset splits:**
  - Training: learn patterns
  - Test: evaluate on unseen data

---

### 4. Hands-On: Text Classification (20-25 min)

**Topics:**

#### **Load and Explore Data**

- SMS spam dataset (2-3 line load)
- Look at examples (2-3 line display)
- Check class balance (2-3 line visualization)

#### **Convert Text to Numbers**

- Bag-of-words: count word occurrences
- Code: 2-3 lines using CountVectorizer
- Visualize: sparse matrix shape

#### **Train a Classifier**

- Logistic regression on word counts
- Code: 2-3 lines (vectorize, fit, predict)
- Show predictions on test examples

#### **Evaluate Performance**

- Accuracy metric
- Confusion matrix (2-3 line visualization)
- Look at errors: what did model get wrong?

#### **Demonstrate Overfitting**

- Train with 1-grams, 2-grams, 3-grams
- Plot: training vs test accuracy
- Show: complex models memorize training data

---

### 5. Hands-On: Simple Language Model (20-25 min)

**Topics:**

#### **What is a Language Model?**

- Predicts next word: P(word | previous words)
- Example: "I love ___" → high P(this), low P(banana)
- Application: text generation, autocomplete

#### **Build a Bigram Model**

- Count word pairs from corpus
- Code: 2-3 lines to build counts dictionary
- Normalize to get probabilities

#### **Generate Text**

- Start with a word
- Sample next word from bigram probabilities
- Code: 2-3 line sampling loop
- Show: generated sentences (often nonsensical!)

#### **Evaluate with Perplexity**

- Perplexity: how "surprised" is model?
- Formula: $PP = P(w_1,...,w_n)^{-1/n}$
- Code: 2-3 lines to compute
- Lower perplexity = better predictions

#### **Fundamental Limitations**

- **Sparsity:** Unseen word pairs get zero probability
- **No context:** Can't use information beyond 1 word back
- **No generalization:** "cat sat" and "dog sat" are unrelated
- **Demonstration:** Show failure cases

**Bridge:** Neural networks solve these problems (next lecture)

---

### 6. Evaluation and Metrics (5-10 min)

**Topics:**

- **Classification:** Accuracy, precision, recall
- **Language models:** Perplexity
- **When metrics mislead:**
  - Imbalanced classes → accuracy looks good but model is bad
  - Example with spam dataset

---

### 7. Summary and Next Steps (5 min)

**Key Takeaways:**

- ML learns patterns from (text, label) pairs
- Classification: convert text → numbers → predict label
- Language modeling: learn P(next word | context)
- Limitations: sparse features, no generalization, limited context

**Next lecture:** Neural networks learn better text representations

---

## Supporting Materials

### Code Structure: Every Cell is 2-3 Lines

**Critical principle:** Each code cell demonstrates ONE concept in 2-3 meaningful lines.

**Examples:**

```python
# Load data (2 lines)
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'soc.religion.christian'])
```

```python
# Vectorize text (2 lines)
from sklearn.feature_extraction.text import CountVectorizer
X = CountVectorizer().fit_transform(data.data)
```

```python
# Train and evaluate (3 lines)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000).fit(X[:800], data.target[:800])
print(f"Accuracy: {model.score(X[800:], data.target[800:]):.2f}")
```

### Notebook Sections

#### **Section 1: Library Basics (3-4 cells)**

- Import numpy, show array creation (2 lines)
- Import sklearn, load sample text data (2 lines)
- Import collections, demonstrate Counter (2 lines)
- Show data shape and first examples (2 lines)

#### **Section 2: Text Classification (8-10 cells)**

1. Load SMS spam dataset (2 lines)
2. Display sample messages (2 lines)
3. Check class distribution (2 lines + plot)
4. Convert text to bag-of-words (2 lines)
5. Print matrix shape (1 line)
6. Train logistic regression (2 lines)
7. Predict on test set (2 lines)
8. Show confusion matrix (2-3 lines)
9. Demonstrate overfitting: train with different n-gram sizes (3 lines per model)
10. Plot training vs test curves (3 lines)

#### **Section 3: Language Model (8-10 cells)**

1. Load simple text corpus (2 lines)
2. Build bigram counts with Counter (3 lines)
3. Show top bigrams (2 lines)
4. Convert counts to probabilities (2-3 lines)
5. Sample next word given context (2-3 lines)
6. Generate sequence with loop (3 lines)
7. Show generated examples (1 line)
8. Compute perplexity on test (3 lines)
9. Demonstrate failure: unseen bigram → zero probability (2 lines)

#### **Section 4: Evaluation (3-4 cells)**

1. Classification metrics from sklearn (2 lines)
2. Show imbalanced dataset problem (3 lines)
3. Perplexity formula and computation (2-3 lines)

### Datasets

**Primary dataset: SMS Spam Collection**

- ~5,500 text messages
- Binary labels (spam/ham)
- Small enough to show overfitting clearly
- Load programmatically (no files needed)

**For language model: Simple text corpus**

- WikiText sample or similar
- Or use SMS data itself
- Just need raw text for bigram counts

### Visualizations (3-4 total)

1. **Class balance bar chart** (2 lines: value_counts + plot)
2. **Confusion matrix heatmap** (3 lines: compute, plot, labels)
3. **Overfitting curve** (3 lines: collect scores, plot train vs test)
4. **Bigram probability distribution** (3 lines: top 10 bigrams as bar chart)

### Mathematical Content (Minimal)

**Only include formulas directly connected to code:**

1. **Bigram probability:** $P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$
   - Show formula, then implement in 2-3 lines

2. **Perplexity:** $PP = P(w_1,...,w_n)^{-1/n}$
   - Show formula, then compute in 2-3 lines

3. **No derivations** - just formulas needed for implementation

### Student Exercises

#### **Exercise 1: Modify the classifier (10 min)**

- Change from 1-grams to 2-grams
- Compare accuracy
- Explain why it changed

#### **Exercise 2: Improve the language model (10 min)**

- Build trigram model instead of bigram
- Generate text and compare quality
- Compute perplexity - did it improve?

#### **Exercise 3: Error analysis (10 min)**

- Find 5 misclassified spam messages
- Read them - why did model fail?
- Suggest one feature that could help

### Recommended Reading (Simplified)

1. **Jurafsky & Martin, Ch. 3** - N-gram Language Models
   - Just section 3.1-3.3 (first 10 pages)

2. **Jurafsky & Martin, Ch. 4** - Naive Bayes Classification
   - Just section 4.1-4.2 (basic classifier)

3. **Scikit-learn text tutorial**
   - Working with text data guide

### Teaching Notes

**Live coding approach:**

- Type each 2-3 line cell live during lecture
- Run immediately to show output
- Ask students: "What do you expect to see?"
- Discuss the output before moving to next cell

**Common discussion points per cell:**

- "Why did we get this shape?"
- "What does this number mean?"
- "What happens if we change this parameter?"
- "Where might this model fail?"

**Pacing:**

- Spend 2-3 minutes per code cell
- Show, run, discuss, move on
- Don't let any cell take >5 minutes
