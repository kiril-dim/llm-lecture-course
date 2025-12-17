# Lecture 12: AI Agents and Tools

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 10 (Prompting), Lecture 11 (RAG)
**Next Lecture:** Course Conclusion

**Lecture Style:** Conceptual with demonstrations. Focus on agent architectures, reasoning patterns, and practical limitations. Students should understand when agents help and when they fail.

---

## Lecture Outline

### 1. From Generation to Action (15-20 min)

- The limitation of pure generation: models can only produce text
- What if models could use tools, access APIs, browse the web?
- The agent paradigm: observe, think, act, observe again
- Brief history: from GPT plugins to modern agent frameworks
- Why 2023-2024 saw an explosion of agent research

### 2. Agent Architecture Fundamentals (20-25 min)

- Core components: LLM brain, tools, memory, orchestration
- The agent loop: perception → reasoning → action → observation
- Tool definitions: function signatures, descriptions, parameter schemas
- How models choose which tool to use
- Parsing model outputs into executable actions
- Error handling and recovery

**Demo:** Simple agent with calculator and search tools

### 3. ReAct: Reasoning and Acting (20-25 min)

- The ReAct pattern: interleave reasoning traces with actions
- Why reasoning before acting improves tool selection
- ReAct prompt structure: Thought → Action → Observation → ...
- Comparison to chain-of-thought: reasoning with grounding
- Strengths and failure modes
- When ReAct helps vs standard prompting

**Demo:** ReAct agent solving a multi-step problem

### 4. Agent Memory (20-25 min)

- The context window limitation
- Short-term memory: conversation history, working context
- Long-term memory: vector stores, summaries, structured storage
- Memory retrieval: what to remember, what to forget
- Episodic vs semantic memory patterns
- Memory management strategies for long-running agents

### 5. Planning and Reflection (20-25 min)

- Planning: decomposing complex tasks into subtasks
- Plan-and-execute patterns
- Reflection: self-critique and improvement
- Reflexion: learning from mistakes
- Self-correction loops: detect errors, adjust approach
- The challenge of knowing when you're wrong

**Demo:** Agent with planning and self-correction

### 6. Tools in Practice (15-20 min)

- Common tool categories: search, code execution, file I/O, APIs
- Tool design principles: clear descriptions, constrained parameters
- The tool selection problem at scale
- Code interpreters: executing generated code safely
- Browser automation and web agents
- Security considerations: sandboxing, permissions, rate limits

### 7. Multi-Agent Systems (15-20 min)

- Why multiple agents? Specialization, debate, collaboration
- Architectures: hierarchical, peer-to-peer, debate
- Agent communication protocols
- Ensemble approaches: multiple agents, one task
- Practical examples: coding assistants, research teams
- Coordination challenges and overhead

### 8. Current Results and Limitations (15-20 min)

- Benchmark performance: SWE-bench, WebArena, GAIA
- Where agents succeed: structured tasks, clear tool boundaries
- Where agents fail: long-horizon planning, error accumulation
- The reliability problem: agents are brittle
- Cost and latency considerations
- The gap between demos and production

### 9. Summary and Course Conclusion (10 min)

**Key takeaways:**
- Agents extend LLMs from generation to action
- ReAct combines reasoning with tool use
- Memory enables longer interactions
- Planning and reflection improve complex tasks
- Current agents are capable but unreliable

**Course arc:**
- Started with ML fundamentals and language modeling
- Built up to transformers and pretraining
- Understood emergent capabilities and alignment
- Learned to deploy locally and prompt effectively
- Grounded models with RAG, extended them with agents

**What's next for the field:**
- More reliable agents
- Better planning and reasoning
- Tighter integration with external systems
- The path toward more capable AI systems

---

## Supporting Materials

### Code Demos
1. Basic tool-using agent
2. ReAct implementation
3. Agent with memory
4. Planning and reflection patterns
5. Simple multi-agent setup

### Key Visualizations
- Agent architecture diagram
- ReAct trace visualization
- Memory architecture patterns
- Multi-agent coordination flows

### Exercises
1. Build an agent with 3-4 tools, test on multi-step tasks
2. Implement ReAct pattern, compare to direct tool calling
3. Add memory to an agent, test on long conversations
4. Design a two-agent system for a collaborative task
5. Analyze agent failure modes on complex problems

### Key Papers
- Yao et al. (2022) — ReAct
- Shinn et al. (2023) — Reflexion
- Park et al. (2023) — Generative Agents
- Wang et al. (2024) — Survey of LLM-based Agents
