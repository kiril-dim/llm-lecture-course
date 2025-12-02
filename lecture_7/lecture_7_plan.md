# Lecture 7: Emergent Capabilities at Scale

## Course Information

**Duration:** 2-2.5 hours  
**Prerequisites:** Lecture 6 (Foundation Models, Scaling Laws)  
**Next Lecture:** Alignment and RLHF

---

## Lecture Outline

### 1. Introduction: What Are Emergent Capabilities? (10-15 min)

**Topics:**

- Recap: Scaling laws from Lecture 6
- **Definition of emergence:**
  - Capabilities not explicitly trained for
  - Qualitative jumps, not just quantitative improvements
  - "Phase transitions" at certain scale thresholds
- **Examples that surprised researchers:**
  - Multi-step reasoning without explicit training
  - Code generation from natural language
  - Answering questions in languages not in training data
  - Few-shot learning from demonstrations
- **The central mystery:**
  - Why does predicting next token lead to reasoning?
  - Connection to compression and world modeling
- **Lecture roadmap:** From zero-shot to reasoning

---

### 2. Zero-Shot and Few-Shot Learning (25-30 min)

**Topics:**

#### **Traditional ML paradigm (recap)**

- Train on task-specific labeled data
- One model per task
- Fine-tuning required for new tasks

#### **The new paradigm: In-Context Learning**

- No gradient updates
- Learn from prompt at inference time
- Same model, many tasks

#### **Zero-Shot Learning**

- Task description only, no examples
- Example: "Translate to French: Hello" → "Bonjour"
- **When it emerged:** GPT-3 (175B parameters) showed reliable zero-shot
- **What enables it:**
  - Instruction understanding from pretraining
  - Task generalization
  - World knowledge

#### **Few-Shot Learning**

- Provide k examples in the prompt (typically 1-10)
- Model infers pattern and applies to new example
- **Classic example:**

```
  English: Hello, French: Bonjour
  English: Goodbye, French: Au revoir
  English: Thank you, French: [model completes]
```

- **Performance scaling:**
  - Small models (<1B): minimal few-shot ability
  - Medium models (1-10B): some few-shot, inconsistent
  - Large models (>10B): strong few-shot, improves with k
  - Very large models (>100B): approaches fine-tuned performance

#### **Demonstration: Few-shot prompting**

- Sentiment classification with 0, 1, 5, 10 examples
- Show performance improvement with examples
- Compare model sizes on same task

**Code example:** Implement few-shot prompting, vary k, measure accuracy

---

### 3. In-Context Learning Mechanics (25-30 min)

**Topics:**

#### **What is happening during inference?**

- **Not gradient descent:** weights are frozen
- **Forward pass only:** attention mechanism processes prompt + query
- **Analogy:** Model as meta-learner or algorithm learning algorithm

#### **Current theories:**

**Theory 1: Implicit fine-tuning**

- Attention updates create gradient-like changes in representations
- Transformer implements optimization algorithm
- Evidence: ICL mimics fine-tuning behavior

**Theory 2: Task recognition**

- Model saw similar patterns in pretraining
- Recognizes task type from examples
- Retrieves relevant "circuits" or knowledge

**Theory 3: Bayesian inference**

- Model learns distribution over tasks during pretraining
- Few-shot examples update posterior over tasks
- Predicts based on most likely task

#### **Key research findings:**

- More examples generally help (up to context limit)
- Example order matters (recency bias)
- Example quality > quantity
- Instruction format matters significantly
- Models can learn "how to learn" from pretraining

#### **What affects ICL performance:**

- Model scale (larger = better ICL)
- Pretraining data diversity
- Example selection and ordering
- Prompt template design
- Task similarity to pretraining distribution

#### **Limitations and failure modes:**

- Limited by context window (can't fit many examples)
- Inconsistent across tasks
- Sensitive to prompt wording
- Poor at tasks requiring procedural knowledge
- Can't update internal knowledge

**Visualizations:**

- Attention patterns during few-shot learning
- Performance vs number of examples
- Comparison: ICL vs fine-tuning curves

---

### 4. Emergent Reasoning Capabilities (30-35 min)

**Topics:**

#### **Mathematical Reasoning**

**Arithmetic:**

- Small models: fail on basic addition
- Medium models: single-step arithmetic
- Large models: multi-digit arithmetic, word problems

**GSM8K benchmark (grade school math):**

- 8,000 math word problems
- Requires multi-step reasoning
- **Performance by scale:**
  - GPT-2 (1.5B): ~0% accuracy
  - GPT-3 (175B): ~35% accuracy
  - PaLM (540B): ~56% accuracy
  - GPT-4: ~92% accuracy
- Sharp emergence around 100B parameters

**Code example:** Test models on math problems, show scaling

#### **Logical Reasoning**

- Deductive reasoning tasks
- Syllogisms and logic puzzles
- When it emerges: >10B parameters, quality improves with scale
- Still not perfect: failure on complex multi-hop reasoning

#### **Commonsense Reasoning**

- Physical intuition: "If I drop a ball, what happens?"
- Social reasoning: understanding human behavior
- Temporal reasoning: before/after relationships
- **Benchmarks:** PIQA, HellaSwag, WinoGrande
- Gradual improvement with scale, not sharp emergence

#### **Code Generation and Understanding**

**Evolution with scale:**

- Small models: syntax errors, incomplete code
- Medium models: simple functions, common patterns
- Large models: complex algorithms, multi-file projects
- **HumanEval benchmark:**
  - Codex (12B): ~28% pass@1
  - GPT-3.5: ~48% pass@1
  - GPT-4: ~67% pass@1

**What enables code understanding:**

- Code in pretraining data
- Programming as structured reasoning
- Compression objective favors generalizable patterns

**Demonstration:** Generate increasingly complex code with different model sizes

#### **Multilingual Capabilities**

**Zero-shot cross-lingual transfer:**

- Trained mostly on English
- Performs on languages with minimal training data
- Quality correlates with amount of training data per language

**Translation without parallel data:**

- Not trained on translation pairs
- Emerges from monolingual text
- Why it works: shared concepts across languages

**Limitations:**

- English-centric bias
- Poorer performance on low-resource languages
- Cultural knowledge gaps

#### **Instruction Following**

**Evolution:**

- Early models: ignore instructions, continue text
- Larger models: follow simple instructions
- Largest models: complex multi-step instructions

**What improved:**

- Understanding task from natural language
- Differentiating instruction vs content
- Following constraints and formatting requirements

**Example progression:**

```
Small model:
Instruction: "List 3 capitals in Europe"
Output: "in Europe. The weather is..."

Large model:
Instruction: "List 3 capitals in Europe"
Output: "1. Paris, France 2. Berlin, Germany 3. Rome, Italy"
```

---

### 5. Measuring Emergence (20-25 min)

**Topics:**

#### **Standard Benchmarks**

**MMLU (Massive Multitask Language Understanding):**

- 57 subjects from humanities to STEM
- Measures breadth of knowledge
- Clear scaling trends: larger models score higher
- Sharp improvement around 50-100B parameters

**BIG-Bench:**

- 200+ diverse tasks
- Specifically designed to find emergence
- Some tasks show sharp transitions

**GSM8K, HellaSwag, others:**

- Task-specific benchmarks
- Show different emergence patterns

#### **Sharp vs Smooth Emergence Debate**

**Sharp emergence view:**

- Some capabilities appear suddenly at scale
- Phase transition analogy
- Examples: arithmetic, instruction following

**Smooth emergence view:**

- May be artifact of measurement
- Different metrics show different patterns
- Emergent = unpredictable, not necessarily sudden

**Recent findings:**

- Choice of metric affects perception of emergence
- Some "sharp" transitions smooth out with better metrics
- Some capabilities genuinely appear suddenly

#### **Evaluation Challenges**

**Prompt sensitivity:**

- Performance varies with prompt wording
- Hard to compare across models
- Need for standardized evaluation

**Contamination:**

- Benchmarks may be in training data
- Inflates apparent performance
- Major concern for large web-scraped datasets

**Saturation:**

- Existing benchmarks too easy for largest models
- Need harder benchmarks
- Moving target problem

**Code demo:** Run same model on benchmark with different prompts, show variance

---

### 6. What Doesn't Emerge (15-20 min)

**Topics:**

#### **Persistent Limitations**

**Factual accuracy:**

- Models still hallucinate
- Don't improve monotonically with scale
- Need external knowledge sources

**Consistency:**

- Can contradict themselves
- Context-dependent responses
- No stable "beliefs"

**Planning and long-horizon reasoning:**

- Multi-step plans often fail
- Lose track in long reasoning chains
- Sequential errors accumulate

**Calibration:**

- Overconfident on wrong answers
- Poor uncertainty estimates
- Doesn't improve proportionally with scale

#### **Inverse Scaling**

- Some tasks get WORSE with scale
- Examples:
  - Repetitive tasks (models generate too much)
  - Sycophancy (agree with user even when wrong)
  - Certain reasoning traps
- **Why:** Larger models better at pattern matching, including problematic patterns

#### **U-Shaped Scaling Curves**

- Performance improves, then degrades, then improves again
- Medium models worst at some tasks
- Hypothesis: insufficient capacity to overcome misleading patterns

**Visualization:** Show inverse scaling and U-shaped curves on specific tasks

---

### 7. Theories of Why Emergence Happens (15-20 min)

**Topics:**

#### **Compression and World Modeling**

- Next-token prediction requires world model
- Compression favors generalizable patterns
- Larger models = better compression = better world models

#### **Circuit Formation**

- Neural networks form specialized "circuits" for subtasks
- Larger models have more circuits
- Composition of circuits enables complex behaviors

#### **Grokking at Scale**

- Models may "grok" (suddenly understand) at scale
- Transition from memorization to generalization
- Similar to small model grokking, but at larger scale

#### **Emergent Optimization**

- Transformers can implement optimization algorithms
- ICL as learned optimization
- Enables adaptation without gradients

#### **The Scaling Hypothesis**

- Intelligence may be primarily about scale
- Architecture and training objective matter less
- Controversial: some argue fundamentally new ideas needed

**Discussion:** Is scale enough? What's missing?

---

### 8. Implications and Future Directions (10-15 min)

**Topics:**

#### **For AI Development**

- Bigger models likely more capable
- But: diminishing returns, efficiency concerns
- Need: better evaluation, safety research

#### **For AI Safety**

- Unpredictable capabilities complicate safety
- Need to anticipate what emerges
- Alignment becomes more critical (Lecture 8 preview)

#### **For Applications**

- Generalist models vs specialist models
- When to use LLMs vs fine-tuned models
- Cost-benefit of scale

#### **Open Questions**

- Will emergence continue indefinitely?
- What capabilities remain out of reach?
- Can we predict emergence before building?

---

### 9. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- Scale unlocks qualitatively new capabilities
- In-context learning enables flexible task performance
- Reasoning emerges from next-token prediction
- Not everything improves with scale
- Emergence is powerful but unpredictable

**What's next:**

- Powerful capabilities need alignment
- How do we make models helpful, harmless, honest?
- **Lecture 8:** RLHF and the alignment problem

---

## Supporting Materials

### Code Examples

1. **Few-shot prompting demonstration**
   - Implement k-shot learning for classification
   - Vary k from 0 to 10
   - Measure accuracy vs number of examples
   - Compare across model sizes (if multiple APIs available)

2. **Zero-shot task performance**
   - Test model on tasks with no examples
   - Translation, summarization, QA
   - Show what works and what doesn't

3. **Math reasoning evaluation**
   - Test on GSM8K problems
   - Compare small vs large models
   - Analyze error patterns

4. **Prompt sensitivity analysis**
   - Same task, different prompt wordings
   - Measure variance in performance
   - Demonstrate brittleness

5. **Benchmark evaluation script**
   - Load standard benchmark (MMLU subset)
   - Evaluate model performance
   - Compare with published results

### Mathematical Derivations

1. **In-context learning as Bayesian inference**
   - Formal model of task inference
   - Posterior update with examples
   - Connection to meta-learning

2. **Scaling law extensions**
   - Predicting capability emergence
   - Perplexity vs task performance relationship
   - Limitations of predictions

3. **Information theory perspective**
   - Compression and generalization
   - Why compression implies understanding
   - Minimum description length principle

### Visualizations

1. **Emergence curves**
   - Performance vs model size for various tasks
   - Sharp vs smooth emergence examples
   - Different benchmarks on same plot

2. **Few-shot scaling**
   - Accuracy vs number of examples
   - Compare model sizes
   - Diminishing returns visualization

3. **Capability map**
   - Which capabilities emerge at which scale
   - Timeline/threshold visualization
   - Color-coded by capability type

4. **Attention patterns in ICL**
   - How model attends to few-shot examples
   - Pattern replication across examples
   - Comparison: with vs without examples

5. **Inverse scaling examples**
   - Tasks that get worse with scale
   - U-shaped curves
   - Explanations for phenomenon

6. **Benchmark comparison table**
   - Model size vs performance on MMLU, GSM8K, etc.
   - Highlight emergence thresholds
   - Include training compute

### Datasets to Use

1. **GSM8K (Grade School Math)**
   - 8,000 math word problems
   - Test reasoning capabilities
   - Clear scaling trends

2. **MMLU (Massive Multitask Language Understanding)**
   - 57 subjects
   - Broad knowledge evaluation
   - Standard for measuring emergence

3. **BIG-Bench (subset)**
   - Diverse tasks
   - Some show sharp emergence
   - Use representative subset for class

4. **HumanEval**
   - Code generation benchmark
   - 164 programming problems
   - Demonstrates code understanding emergence

5. **Custom few-shot datasets**
   - Simple classification tasks
   - For demonstrating ICL mechanics
   - Students can create their own

### Student Exercises

#### Exercise 1: Few-shot learning exploration

- Design few-shot prompts for classification task
- Test with 0, 1, 3, 5, 10 examples
- Analyze: How does performance change?
- Experiment with example selection and ordering
- Document what works and what doesn't

#### Exercise 2: Capability testing

- Choose an emergent capability (math, reasoning, code)
- Design evaluation set (10-20 problems)
- Test multiple models (if API access available)
- Analyze failure modes
- Identify scale threshold where capability appears

#### Exercise 3: Prompt engineering for emergence

- Take a task where model initially fails
- Iteratively improve prompt
- Try: instructions, examples, formatting, chain-of-thought
- Document improvement trajectory
- Reflect: Is this emergence or prompt engineering?

#### Exercise 4: Benchmark deep dive

- Analyze MMLU or GSM8K in detail
- Review actual questions and model responses
- Categorize error types
- Compare human vs model performance
- Propose improvements to evaluation

### Recommended Reading

#### Foundational Papers

1. **"Language Models are Few-Shot Learners"** - Brown et al. (2020)
   - GPT-3 paper
   - Introduced in-context learning at scale
   - Key results on emergence

2. **"Emergent Abilities of Large Language Models"** - Wei et al. (2022)
   - Comprehensive survey of emergence
   - Sharp emergence on many tasks
   - Scaling curves and thresholds

3. **"Inverse Scaling Can Become U-Shaped"** - McKenzie et al. (2023)
   - Some tasks get worse then better with scale
   - Challenges simple scaling hypothesis

4. **"Are Emergent Abilities of Large Language Models a Mirage?"** - Schaeffer et al. (2023)
   - Argues emergence may be measurement artifact
   - Metric choice affects perception
   - Important counterpoint

5. **"What Can Transformers Learn In-Context? A Case Study of Simple Function Classes"** - Garg et al. (2022)
   - Theoretical analysis of ICL
   - Transformers can implement learning algorithms
   - Explains mechanism

#### Understanding In-Context Learning

1. **"Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"** - Min et al. (2022)
   - Example content vs format matters
   - Ground truth labels sometimes unnecessary
   - Challenges assumptions about ICL

2. **"Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers"** - Dai et al. (2022)
   - ICL as implicit optimization
   - Connection to meta-learning
   - Mechanistic explanation

3. **"In-context Learning and Induction Heads"** - Olsson et al. (2022)
   - Circuit-level analysis
   - Induction heads implement ICL
   - Anthropic's mechanistic interpretability work

#### Scaling and Capabilities

1. **"Training Compute-Optimal Large Language Models"** - Hoffmann et al. (2022)
   - Chinchilla scaling laws (from Lecture 6)
   - Relation to emergent capabilities
   - Optimal model and data size

2. **"Scaling Laws for Neural Language Models"** - Kaplan et al. (2020)
   - Original scaling laws paper
   - Loss vs scale relationships
   - Extrapolation limits

3. **"Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models"** - Srivastava et al. (2022)
   - BIG-Bench benchmark
   - Diverse tasks for measuring emergence
   - Results across model scales

#### Specific Capabilities

1. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** - Wei et al. (2022)
   - CoT improves reasoning
   - Emerges at scale
   - Preview for Lecture 10

2. **"Solving Quantitative Reasoning Problems with Language Models"** - Lewkowycz et al. (2022)
   - Mathematical reasoning at scale
   - Minerva model
   - STEM capabilities

3. **"Competition-Level Code Generation with AlphaCode"** - Li et al. (2022)
   - Code generation capabilities
   - Performance at scale
   - DeepMind's approach

#### Textbooks and Surveys

1. **"A Survey of Large Language Models"** - Zhao et al. (2023)
   - Comprehensive recent survey
   - Covers emergence in detail
   - Good overview of field

2. **"Emergent Abilities in AI: Are We Chasing a Myth?"** - Blog post by Stanford HAI
   - Accessible discussion
   - Current debates
   - Multiple perspectives

#### Online Resources

1. **OpenAI Blog**
   - GPT-3, GPT-4 capability posts
   - Examples of emergent abilities
   - Case studies

2. **Anthropic's Research**
   - In-context learning mechanisms
   - Constitutional AI (preview for Lecture 8)
   - Interpretability work

3. **BIG-Bench Project**
   - Benchmark details
   - Leaderboards
   - Task examples

4. **Stanford CS324 - Large Language Models**
   - Lecture notes on capabilities
   - Up-to-date course material

### Additional Materials

#### Interactive Demos

- Few-shot prompting playground
- Compare model sizes on same tasks
- Benchmark visualization tools
- Emergence threshold calculator

#### Discussion Questions

- Is emergence truly unpredictable or just not yet understood?
- How do we design evaluations for capabilities we can't anticipate?
- What are the safety implications of unpredictable emergence?
- Will scaling continue to produce new capabilities indefinitely?
- Are current emergent abilities sufficient for AGI or are fundamental new ideas needed?
- How should we balance scaling existing models vs developing new architectures?

#### Advanced Topics (Brief Mentions)

- Multi-modal emergence (vision + language)
- Emergent optimization algorithms in transformers
- Circuit formation and pruning during training
- Grokking phenomena at different scales
- Meta-learning vs in-context learning distinctions
- Theoretical limits of emergence

#### Lab Session Ideas

- Design and test few-shot prompts
- Reproduce emergence results from papers (on smaller scale)
- Capability testing competition
- Analyze model behavior on edge cases
- Create visualization of benchmark results
- Discussion: predict next emergent capabilities
