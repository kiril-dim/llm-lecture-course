# Lecture 10: Advanced Prompting and Reasoning Models

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 7 (Emergent Capabilities), Lecture 8 (Alignment/RLHF), Lecture 9 (Local LLMs)
**Next Lecture:** Hallucinations and RAG

**Lecture Style:** Narrative-driven with practical demonstrations. Tell the story of how prompting techniques evolved, the key discoveries and papers, then hands-on experimentation with Ollama. Students should understand both the intellectual history and practical application.

---

## Lecture Outline

### 1. Introduction: The Elicitation Problem (10-15 min)

**Core message:** Models have capabilities (L7), alignment shapes behavior (L8), but how do we reliably access those capabilities?

**Topics:**

- Recap: same model weights, dramatically different results based on prompt
- Prompting as "programming" without changing weights
- The discovery that prompting matters enormously

**Key example to show:**

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

Standard: "11" (often wrong, or right by luck)
With "Let's think step by step": Correct reasoning shown
```

**Bridge:** How did we discover these techniques? The evolution story.

---

### 2. The Evolution of Prompting: A Brief History (20-25 min)

**Style:** Intellectual history with landmark papers

**The Timeline:**

#### **2020: Few-Shot Learning Emerges (GPT-3)**

- Brown et al., "Language Models are Few-Shot Learners"
- Discovery: examples in prompt → model learns pattern
- Shocked the field: no fine-tuning needed for many tasks
- Established the in-context learning paradigm

#### **2022: Chain of Thought Discovery**

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning"
- Key insight: making models show their work improves accuracy
- Emergence: only works at scale (>100B parameters)
- Simple phrase unlocks reasoning

- Kojima et al., "Large Language Models are Zero-Shot Reasoners"
- "Let's think step by step" - surprisingly effective
- Zero-shot CoT: no examples needed, just the phrase

#### **2022-2023: Refinements and Variations**

- Wang et al., "Self-Consistency" - majority voting over multiple CoT samples
- Yao et al., "Tree of Thoughts" - search through reasoning paths
- Many variations: Plan-and-Solve, Least-to-Most, etc.

#### **2023-2024: The Reasoning Model Shift**

- Lightman et al., "Let's Verify Step by Step" - process supervision
- Key insight: reward each step, not just final answer
- Foundation for reasoning models

#### **2024-2025: Reasoning Models**

- OpenAI o1/o3: test-time compute scaling
- DeepSeek R1: open reasoning model
- New paradigm: more thinking time = better results

**Discussion:** Why did it take so long to discover these? What does this say about how we understand LLMs?

---

### 3. Prompt Engineering Fundamentals (15-20 min)

**Core techniques with concrete examples:**

#### **Clarity and Specificity**

- Vague vs specific instructions
- Output format control
- The importance of examples (few-shot)

#### **Structural Techniques**

- Delimiters for separating content (```, ###, XML tags)
- Instruction positioning (end often works better)
- Role/persona assignment

#### **Common Pitfalls**

- "Don't do X" often backfires
- Over-complicated prompts
- Ignoring output format

**Practical focus:** These basics matter for everything that follows

---

### 4. Chain of Thought Deep Dive (25-30 min)

**The core technique every practitioner needs**

#### **Why CoT Works**

- LLMs have no hidden computation - only predict next token
- Complex reasoning requires explicit intermediate steps
- CoT forces the model to "think out loud"

#### **Zero-Shot CoT**

- Magic phrase: "Let's think step by step"
- When it helps: multi-step problems, math, logic
- When it doesn't help: simple factual recall

#### **Few-Shot CoT**

- Provide examples with reasoning chains
- Higher quality than zero-shot for specific domains
- Example quality matters enormously

#### **Self-Consistency (Majority Voting)**

- Generate N reasoning chains with temperature > 0
- Extract answer from each
- Take majority vote
- Typical improvement: +5-15% accuracy
- Trade-off: N times the cost

#### **When to Use CoT**

- Multi-step reasoning required
- Math word problems
- Logic puzzles
- Questions with misleading surface features

**Key papers to reference:**
- Wei et al. 2022 (original CoT)
- Kojima et al. 2022 (zero-shot CoT)
- Wang et al. 2022 (self-consistency)

---

### 5. Beyond Linear Chains: Tree of Thought (15-20 min)

**Brief coverage - the concept matters more than implementation details**

#### **The Limitation of Linear CoT**

- Single path through reasoning
- If first step is wrong, can't recover
- No exploration or backtracking

#### **ToT Concept**

- Generate multiple possible next steps
- Evaluate which are promising
- Search through reasoning tree
- Prune bad branches, explore good ones

#### **Classic Example: Game of 24**

- Use 4 numbers with +, -, *, / to make 24
- Multiple valid starting moves
- ToT explores alternatives, backtracks when stuck

#### **Practical Reality**

- Expensive: many LLM calls per problem
- Rarely used in production
- Important conceptually: reasoning as search
- Foreshadows how reasoning models work internally

**Key paper:** Yao et al. 2023, "Tree of Thoughts"

---

### 6. The Reasoning Model Revolution (30-35 min)

**The paradigm shift from prompting to trained reasoning**

#### **What Changed**

- Old paradigm: prompt engineering to elicit reasoning
- New paradigm: models trained specifically for reasoning
- Key innovation: test-time compute scaling

#### **Test-Time Compute Scaling**

- Traditional: more training compute → better model
- New: more inference compute → better reasoning
- Model can "think longer" on hard problems
- Demonstrated by o1: accuracy improves with thinking time

#### **Process Supervision vs Outcome Supervision**

- Outcome supervision (standard RLHF): reward final answer only
- Process supervision: reward each reasoning step
- Key insight from "Let's Verify Step by Step" (Lightman et al.)
- Process supervision catches "lucky" correct answers with bad reasoning

#### **How Reasoning Models Work (Conceptually)**

1. Extended thinking phase (hidden or visible)
2. Self-verification and backtracking
3. Multiple approaches tried internally
4. Answer synthesis

#### **The Current Landscape**

**OpenAI o1/o3:**
- Closed, hidden reasoning
- Seconds to minutes per response
- Strong on math, coding, science

**DeepSeek R1:**
- Open weights
- Visible reasoning traces
- Competitive performance
- Important for understanding the approach

**Claude Extended Thinking:**
- Configurable thinking depth
- Visible reasoning option

#### **When to Use Reasoning Models**

- Complex multi-step problems
- High-stakes decisions where accuracy matters
- Math, coding, scientific reasoning
- NOT for simple queries, creative writing, conversation

#### **Cost-Benefit Reality**

| Approach | Latency | Cost | Best For |
|----------|---------|------|----------|
| Standard model | ~1s | 1x | Most tasks |
| Standard + CoT | ~2-3s | 1x | Reasoning on budget |
| Reasoning model | 10s-5min | 5-20x | Critical reasoning |

**Key papers:**
- Lightman et al. 2023, "Let's Verify Step by Step"
- OpenAI o1 System Card (2024)
- DeepSeek R1 paper (2025)

---

### 7. Practical Exercises with Ollama (25-30 min)

**Hands-on demonstrations using local models**

#### **Exercise 1: CoT Comparison**

Using Ollama with a 7B model (e.g., Mistral, Llama 3):

- Same math problem with and without "Let's think step by step"
- Run 10 times each, compare accuracy
- Observe the reasoning chains

```bash
ollama run mistral "Roger has 5 tennis balls..."
ollama run mistral "Roger has 5 tennis balls... Let's think step by step."
```

#### **Exercise 2: Few-Shot Prompting**

- Classification task (e.g., sentiment)
- Compare 0-shot, 1-shot, 3-shot, 5-shot
- Measure effect of example quality and order

#### **Exercise 3: Self-Consistency Demo**

- Generate 5 CoT responses for same problem
- Extract answers, take majority vote
- Compare to single-shot accuracy

#### **Exercise 4: Prompt Engineering Comparison**

- Same task, different prompt formulations
- Measure which variations work best
- Document findings

#### **Exercise 5: Reasoning Model vs Standard (if API available)**

- Compare local model + CoT vs reasoning model API
- Identify complexity threshold where reasoning model wins
- Calculate cost per correct answer

---

### 8. Practical Decision Framework (10 min)

**When to use what:**

```
Simple factual query → Zero-shot, clear prompt
Need specific format → Add format instructions + examples
Reasoning required → Add CoT
CoT not reliable enough → Self-consistency (majority vote)
Critical accuracy, complex task → Reasoning model
```

**Key principles:**

1. Start simple, add complexity only when needed
2. Measure on representative examples
3. Consider cost: latency, tokens, API costs
4. CoT is free improvement for reasoning tasks
5. Reasoning models are expensive but powerful

---

### 9. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- Prompting evolved from few-shot (2020) → CoT (2022) → reasoning models (2024)
- CoT: "Let's think step by step" unlocks reasoning
- Self-consistency: majority voting improves accuracy
- Reasoning models: test-time compute scaling, process supervision
- Match technique to task complexity and cost constraints

**What's next:**

- Even with good prompting, models hallucinate
- They lack access to current/private information
- **Lecture 11:** RAG - grounding models in external knowledge
  - Why hallucinations happen
  - Semantic search and vector databases
  - Retrieval-augmented generation

---

## Supporting Materials

### Code Examples for Notebook

1. **CoT comparison script**
   - Same problem with/without CoT
   - Multiple runs for statistical significance
   - Accuracy measurement

2. **Few-shot prompt builder**
   - Easy example addition
   - Test different k values
   - Measure performance curves

3. **Self-consistency implementation**
   - Generate N samples
   - Parse answers
   - Majority voting logic

4. **Prompt A/B testing harness**
   - Compare prompt variations
   - Statistical significance testing

### Key Visualizations

1. **Evolution timeline** - From GPT-3 few-shot to reasoning models
2. **CoT accuracy curves** - Performance by model size
3. **Self-consistency** - Accuracy vs number of samples
4. **Decision flowchart** - When to use which technique

### Recommended Reading

#### Foundational Papers (Assign 2-3)

1. **"Chain-of-Thought Prompting Elicits Reasoning"** - Wei et al. (2022)
   - The original CoT paper, essential reading

2. **"Large Language Models are Zero-Shot Reasoners"** - Kojima et al. (2022)
   - Zero-shot CoT discovery

3. **"Let's Verify Step by Step"** - Lightman et al. (2023)
   - Process supervision, foundation for reasoning models

#### Additional Reading

- Wang et al. 2022, "Self-Consistency" - majority voting
- Yao et al. 2023, "Tree of Thoughts" - search-based reasoning
- OpenAI o1 System Card - reasoning model overview
- DeepSeek R1 paper - open reasoning model

#### Surveys

- "The Prompt Report" - Schulhoff et al. (2024) - comprehensive prompting survey

### Student Exercises

#### Exercise 1: CoT evaluation
- Choose 20 math word problems
- Compare zero-shot vs CoT on local model
- Report accuracy difference
- Analyze failure modes

#### Exercise 2: Prompt engineering
- Take classification task
- Design 5 different prompts
- Measure accuracy on test set
- Document what works

#### Exercise 3: Self-consistency cost-benefit
- Implement majority voting
- Plot accuracy vs number of samples
- Calculate: at what cost is the accuracy gain worth it?

#### Exercise 4: When does reasoning help?
- Collect problems of varying difficulty
- Test standard model, CoT, reasoning model (if available)
- Identify crossover points

### Discussion Questions

- Is CoT "real" reasoning or sophisticated pattern matching?
- Why did it take until 2022 to discover CoT?
- Should reasoning traces be hidden or visible?
- Will prompting techniques become obsolete as models improve?
- Is test-time compute scaling economically sustainable?
