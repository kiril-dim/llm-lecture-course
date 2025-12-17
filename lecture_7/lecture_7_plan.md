# Lecture 7: Emergent Capabilities at Scale

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 6 (Foundation Models, Pretraining Data, Scaling Laws)
**Next Lecture:** Alignment and RLHF

**Lecture Style:** Discussion-focused with real examples of model behaviors. Minimal code — emphasis on understanding phenomena through curated examples, visualizations of published results, and critical analysis. Prompt engineering techniques are deferred to Lecture 10.

---

## Lecture Outline

### 1. Introduction: What Are Emergent Capabilities? (10-15 min)

**Style:** Conceptual framing with concrete examples

**Topics:**

- Recap: Lecture 6 — scaling laws, data curation, training at scale
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

**Format:** Markdown exposition with example model outputs (no code required)

---

### 2. Zero-Shot and Few-Shot Learning (20-25 min)

**Style:** Show real model outputs, compare behaviors

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
- Example outputs showing what models can/cannot do zero-shot
- **When it emerged:** GPT-3 (175B parameters) showed reliable zero-shot
- **What enables it:**
  - Instruction understanding from pretraining
  - Task generalization
  - World knowledge

#### **Few-Shot Learning**

- Provide k examples in the prompt (typically 1-10)
- Model infers pattern and applies to new example
- **Show real examples** of few-shot success and failure
- **Performance scaling (from published results):**
  - Small models (<1B): minimal few-shot ability
  - Medium models (1-10B): some few-shot, inconsistent
  - Large models (>10B): strong few-shot, improves with k
  - Very large models (>100B): approaches fine-tuned performance

**Format:** Tables of example inputs/outputs, static visualizations of scaling curves from papers

---

### 3. In-Context Learning Mechanics (20-25 min)

**Style:** Theoretical discussion with diagrams

**Topics:**

#### **What is happening during inference?**

- **Not gradient descent:** weights are frozen
- **Forward pass only:** attention mechanism processes prompt + query
- **Analogy:** Model as meta-learner or algorithm learning algorithm

#### **Current theories (discussion-focused):**

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

#### **Limitations and failure modes:**

- Limited by context window (can't fit many examples)
- Inconsistent across tasks
- Sensitive to prompt wording (defer details to Lecture 10)
- Poor at tasks requiring procedural knowledge
- Can't update internal knowledge

**Format:** Diagrams, markdown tables comparing theories, minimal or no code

---

### 4. Emergent Reasoning Capabilities (25-30 min)

**Style:** Showcase concrete examples of reasoning, discuss what works and what fails

**Topics:**

#### **Mathematical Reasoning**

**Show example problems and model outputs:**

- Simple arithmetic → complex word problems
- GSM8K benchmark examples with model responses
- **Performance by scale (table from published results):**
  - GPT-2 (1.5B): ~0% accuracy
  - GPT-3 (175B): ~35% accuracy
  - PaLM (540B): ~56% accuracy
  - GPT-4: ~92% accuracy
- Discussion: Sharp emergence around 100B parameters

#### **Logical Reasoning**

- Deductive reasoning examples
- Syllogisms: what models get right and wrong
- Still not perfect: failure cases on complex multi-hop reasoning

#### **Commonsense Reasoning**

- Examples: physical intuition, social reasoning, temporal reasoning
- **Benchmarks:** PIQA, HellaSwag, WinoGrande (brief descriptions)
- Gradual improvement with scale, not sharp emergence

#### **Code Generation and Understanding**

**Show evolution of code generation quality:**

- HumanEval benchmark results (table)
- Example: same coding problem across different model sizes
- What enables code understanding (discussion)

#### **Multilingual Capabilities**

- Zero-shot cross-lingual transfer examples
- Translation without parallel data
- Limitations: English-centric bias, low-resource languages

#### **Instruction Following**

**Before/after examples:**

- Early models: ignore instructions, continue text
- Larger models: follow complex multi-step instructions

**Format:** Rich markdown with curated examples, benchmark tables, one optional code cell to display examples interactively

---

### 5. Measuring Emergence (15-20 min)

**Style:** Critical analysis of benchmarks and metrics

**Topics:**

#### **Standard Benchmarks (overview)**

**MMLU:**

- What it measures, how it's structured
- Scaling trends visualization

**BIG-Bench:**

- 200+ diverse tasks designed to find emergence
- Some show sharp transitions, some don't

#### **Sharp vs Smooth Emergence Debate**

**Show the controversy:**

- Original emergence curves (Wei et al.)
- Critique: "Are Emergent Abilities a Mirage?" (Schaeffer et al.)
- Resolution: depends on metric choice

**Key insight:** Emergence may be partly measurement artifact

#### **Evaluation Challenges**

- Prompt sensitivity
- Contamination concerns
- Benchmark saturation

**Format:** Reproduced figures from papers, discussion questions

---

### 6. What Doesn't Emerge (15-20 min)

**Style:** Critical examination with failure examples

**Topics:**

#### **Persistent Limitations**

**Factual accuracy:**

- Hallucination examples
- Don't improve monotonically with scale

**Consistency:**

- Examples of self-contradiction
- Context-dependent responses

**Planning and long-horizon reasoning:**

- Multi-step plan failures
- Where reasoning chains break down

**Calibration:**

- Overconfident wrong answers
- Poor uncertainty estimates

#### **Inverse Scaling**

- Some tasks get WORSE with scale
- **Examples:**
  - Sycophancy (agreeing with incorrect premises)
  - Certain reasoning traps
- **Why:** Larger models better at pattern matching, including problematic patterns

#### **U-Shaped Scaling Curves**

- Performance improves, then degrades, then improves again
- Examples and explanations

**Format:** Tables of failure cases, discussion of why these persist

---

### 7. Theories of Why Emergence Happens (15-20 min)

**Style:** Philosophical discussion of competing explanations

**Topics:**

#### **Compression and World Modeling**

- Next-token prediction requires world model
- Compression favors generalizable patterns
- Discussion: Is this sufficient explanation?

#### **Circuit Formation**

- Neural networks form specialized "circuits"
- Composition enables complex behaviors
- Connection to interpretability research

#### **Grokking at Scale**

- Sudden understanding after extended training
- Transition from memorization to generalization

#### **The Scaling Hypothesis**

- Is intelligence primarily about scale?
- Arguments for and against
- What's missing from pure scaling?

**Discussion questions:** Is scale sufficient? What architectural innovations might help?

**Format:** Conceptual markdown, no code

---

### 8. Implications and Future Directions (10-15 min)

**Style:** Forward-looking discussion

**Topics:**

#### **For AI Development**

- Bigger models likely more capable, but diminishing returns
- Efficiency concerns and alternatives to pure scaling

#### **For AI Safety**

- Unpredictable capabilities complicate safety
- Preview of Lecture 8: Why alignment matters

#### **Open Questions**

- Will emergence continue indefinitely?
- What capabilities remain out of reach?
- Can we predict emergence before building?

**Format:** Discussion-oriented, no code

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
