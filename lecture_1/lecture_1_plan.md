# Lecture 1: Introduction to AI and ML (NLP-Focused)

## Course Information

**Duration:** 2-2.5 hours  
**Prerequisites:** Basic programming (Python), linear algebra, probability  
**Next Lecture:** Language Models and Word Representations

---

## Lecture Outline

### 1. Motivation: AI and ML for Language (10-15 min)

**Topics:**

- What is AI vs ML vs NLP?
- **Language understanding challenges:**
  - Ambiguity, context, semantics
  - Why language is hard for computers
- **Success stories:**
  - Machine translation (Google Translate)
  - Chatbots and assistants (ChatGPT, Claude)
  - Sentiment analysis for business
  - Text summarization
- Course roadmap with NLP focus

---

### 2. Problem Formulation in ML (15-20 min)

**Topics:**

- Formal definition with NLP context
- **Key notation:**
  - Input space $\mathcal{X}$ (text documents, sentences, words)
  - Output space $\mathcal{Y}$ (labels, translations, sentiment)
  - Training examples $(x^{(i)}, y^{(i)})$
  - Hypothesis $h: \mathcal{X} \to \mathcal{Y}$
- **Concrete NLP example:** Spam email classification
  - Input: email text
  - Output: {spam, not spam}
  - What does "learning" mean here?

---

### 3. Core ML Terminology with Text Data (20-25 min)

**Topics:**

- **Features in NLP:**
  - Word counts, n-grams, TF-IDF (preview)
  - Example: representing "I love this movie" as features
- **Labels:**
  - Sentiment (positive/negative)
  - Named entity tags
  - Next word in sequence
- **Dataset splits:**
  - Training: learn patterns in text
  - Validation: tune model hyperparameters
  - Test: final evaluation
- **Code example:** Split IMDB movie reviews dataset
- Why this matters for language models (foreshadowing)

---

### 4. Learning Paradigms with NLP Examples (25-30 min)

#### **Supervised Learning**

**Text Classification**

- Example: Sentiment analysis
- Labels: {positive, negative, neutral}
- Code: Logistic regression on movie reviews

**Sequence Labeling**

- Example: Named Entity Recognition (NER)
- Input: "Apple Inc. is located in California"
- Output: [B-ORG, I-ORG, O, O, O, B-LOC]

**Sequence-to-Sequence**

- Example: Machine translation
- Preview: this leads to transformers (Lecture 5)

#### **Unsupervised Learning**

**Topic Modeling**

- Discovering themes in document collections
- Example: clustering news articles

**Word Embeddings (preview)**

- Learning word representations from text
- Foreshadowing: Word2Vec in Lecture 2

**Language Modeling**

- Learning text patterns without explicit labels
- Preview: this is foundation of LLMs (Lectures 6-7)

#### **Reinforcement Learning**

- Brief mention: RLHF for aligning LLMs
- We'll see this concept later in the course

---

### 5. Generalization in NLP (20-25 min)

**Topics:**

- **Overfitting in text models:**
  - Memorizing training sentences verbatim
  - Not generalizing to new phrasings
  - **Example:** Model that only works on training vocabulary
- **Underfitting:**
  - Bag-of-words model missing word order
  - Too simple representations
- **Demonstration:**
  - Train text classifier with increasing model complexity
  - Show overfitting on small text dataset
  - Learning curves with text data
- **Connection to LLMs:**
  - Why we need massive datasets (Lecture 7)
  - Test set contamination concerns

---

### 6. Evaluation Metrics for NLP (20-25 min)

#### **For Text Classification:**

- Accuracy, Precision, Recall, F1
- **Example:** Sentiment analysis confusion matrix
- When accuracy misleads (imbalanced classes)

#### **For Sequence Tasks:**

- **Token-level accuracy** (NER, POS tagging)
- **BLEU score** (machine translation preview)
- **Perplexity** (language models - Lecture 2)

#### **For Generation:**

- Brief mention: ROUGE, human evaluation
- More in later lectures

---

### 7. NLP-Specific Challenges Preview (10-15 min)

**Topics:**

- **Representing text numerically**
  - Words → vectors (Lecture 2)
  - Tokenization challenges (Lecture 3)
- **Variable-length sequences**
  - Sentences have different lengths
  - How do we handle this? (Neural networks - Lecture 4)
- **Context and meaning**
  - "I'm feeling blue" vs "The sky is blue"
  - Preview: Transformers and attention (Lecture 5)
- **Scale and data**
  - Modern LLMs trained on trillions of tokens
  - Preview: Scaling laws (Lecture 7)

---

### 8. Practical NLP Workflow (10 min)

**Steps:**

1. Define NLP task (classification, generation, etc.)
2. Collect and explore text data
3. Text preprocessing and tokenization
4. Feature extraction or embeddings
5. Split data (careful with time-based data)
6. Choose model
7. Train and validate
8. Evaluate on test set
9. Error analysis (look at misclassified examples)

**Common pitfalls in NLP:**

- Data leakage in text
- Test set contamination (big issue for LLMs)
- Vocabulary mismatch

---

### 9. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- ML fundamentals apply to NLP
- Text brings unique challenges
- **Next lecture:** How do we represent words? Language models and embeddings

---

## Supporting Materials

### Code Examples (all with text data)

1. **Load and explore a text dataset**
   - IMDB movie reviews
   - AG News articles
   - SMS spam dataset
   - Basic statistics: vocabulary size, document lengths, class distribution

2. **Train/test split for text classification**
   - Proper splitting for text data
   - Stratified splits to maintain class balance
   - Handling time-based data

3. **Simple bag-of-words classifier**
   - CountVectorizer and TfidfVectorizer
   - Logistic regression on text features
   - Compare different feature representations

4. **Visualize overfitting on text data**
   - Train models with varying complexity
   - Plot training vs validation curves
   - Show effect of vocabulary size on overfitting

5. **Compute NLP metrics**
   - Classification report for sentiment analysis
   - Confusion matrix visualization
   - Precision-recall curves

### Mathematical Derivations

1. **Cross-entropy loss for classification**
   - Binary cross-entropy: $L = -\frac{1}{n}\sum_{i=1}^{n}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$
   - Multi-class cross-entropy
   - Why this is used in all language models

2. **Empirical Risk Minimization**
   - Loss function: $L(h) = \frac{1}{n}\sum_{i=1}^{n}\ell(h(x^{(i)}), y^{(i)})$
   - Connection to maximum likelihood estimation
   - Why log probability matters for language modeling

3. **Bias-Variance Decomposition**
   - Expected test error = Bias² + Variance + Irreducible Error
   - Intuition in context of text classification

### Visualizations

1. **Text dataset statistics**
   - Vocabulary size distribution
   - Document length distribution (histogram)
   - Class balance (bar chart)
   - Word frequency plots (Zipf's law preview)

2. **Confusion matrix for sentiment analysis**
   - Heatmap showing true vs predicted labels
   - Identify common error patterns

3. **Learning curves showing overfitting on text**
   - Training accuracy vs validation accuracy
   - Effect of training set size
   - Effect of model complexity (n-gram size)

4. **Word cloud of most informative features**
   - Positive sentiment words
   - Negative sentiment words
   - Most discriminative features for classification

5. **Feature representation comparison**
   - Sparse vs dense representations
   - Bag-of-words vs TF-IDF visualization

### Datasets to Use

1. **IMDB Movie Reviews**
   - 50,000 reviews (25k train, 25k test)
   - Binary sentiment classification
   - Good for demonstrating overfitting

2. **AG News**
   - 120,000 news articles
   - 4 classes (World, Sports, Business, Sci/Tech)
   - Multi-class classification example

3. **SMS Spam Collection**
   - ~5,500 messages
   - Binary classification (spam/ham)
   - Small dataset good for showing overfitting

4. **20 Newsgroups** (optional)
   - Topic classification
   - Shows importance of vocabulary

### Student Exercises

#### Exercise 1: Build a spam classifier

- Load SMS spam dataset
- Split into train/validation/test
- Extract features (bag-of-words)
- Train logistic regression
- Evaluate with precision, recall, F1
- Analyze misclassified examples

#### Exercise 2: Analyze errors in sentiment prediction

- Train sentiment classifier on IMDB
- Find examples where model fails
- Categorize types of errors:
  - Sarcasm
  - Negation
  - Context dependency
- Suggest improvements

#### Exercise 3: Compare bag-of-words vs more complex features

- Implement unigrams, bigrams, trigrams
- Compare performance
- Plot learning curves
- Show overfitting with complex features on small data

#### Exercise 4: Dataset split investigation

- Implement different splitting strategies
- Random split
- Time-based split (if applicable)
- Stratified split
- Compare impact on evaluation

### Recommended Reading

#### Foundational Papers

1. **"A Few Useful Things to Know About Machine Learning"** - Pedro Domingos (2012)
   - Excellent overview of ML fundamentals
   - Common pitfalls and practical wisdom

2. **"Natural Language Processing (almost) from Scratch"** - Collobert et al. (2011)
   - Early deep learning for NLP
   - Shows evolution of the field

#### Textbooks

1. **"Speech and Language Processing"** - Jurafsky & Martin (3rd ed.)
   - Chapter 2: Regular Expressions, Text Normalization, Edit Distance
   - Chapter 4: Naive Bayes and Sentiment Classification

2. **"Introduction to Information Retrieval"** - Manning, Raghavan & Schütze
   - Chapter 13: Text Classification and Naive Bayes

3. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - Chapter 1: Introduction (ML fundamentals)

#### Online Resources

1. **Stanford CS229 Lecture Notes**
   - Supervised Learning section
   - Bias-variance tradeoff

2. **Stanford CS224N Lecture 1**
   - Introduction to NLP and Deep Learning

3. **Scikit-learn Documentation**
   - Text feature extraction
   - Model evaluation metrics

### Additional Materials

#### Slide Deck Suggestions

- Use concrete examples throughout
- Include live coding demonstrations
- Show failure cases (overfitting, poor features)
- Interactive polls: "What metric should we use here?"

#### Lab/Tutorial Session

- Jupyter notebook walkthrough
- Students implement spam classifier step-by-step
- Instructor demonstrates common errors
- Q&A on concepts
