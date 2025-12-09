# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Educational repository for a 12-lecture course on Large Language Models (LLMs) and Natural Language Processing, taught in Bulgarian. The course progresses from ML fundamentals to advanced LLM topics.

## Project Structure

```
.
├── course_overview.md              # HIGH-LEVEL COURSE PLAN - lists all 12 lectures with brief descriptions
├── ml-terms-en-bg.md              # TERMINOLOGY DICTIONARY - English-Bulgarian ML/AI term translations
├── lecture_N/
│   ├── lecture_N_plan.md          # Detailed lecture plan (outline, exercises, readings)
│   └── [topic].ipynb               # Jupyter notebook with Bulgarian content and code
├── pyproject.toml                  # uv-managed dependencies
├── .python-version                 # Python 3.13
└── .venv/                          # Virtual environment
```

### Key Files

**`course_overview.md`**: The master reference for course structure. Lists all 12 lectures with 1-2 sentence descriptions of each topic. Check this first to understand the course flow.

**`ml-terms-en-bg.md`**: Reference dictionary for technical terminology. Contains ~100 ML/AI terms with:

- English terms (left column)
- Bulgarian translations (middle column)
- Usage notes (right column, e.g., "formal term", "kept in English", etc.)

When working with Bulgarian content or adding new technical terms, consult this dictionary to maintain consistent terminology.

**Lecture directories**: Each complete lecture has:

1. `lecture_N_plan.md` - Comprehensive instructor guide (2-4K lines)
2. `[topic].ipynb` - Jupyter notebook with actual lecture content

## Language Convention

- **Lecture content**: Bulgarian (Cyrillic script)
- **Code**: Python with English variable/function names
- **Comments**: Can be English or Bulgarian
- **Technical terms**: Mixed per ml-terms-en-bg.md conventions
- **Math notation**: Latin script (X, y, W, θ, etc.)

## Development Commands

### Setup

```bash
# Install dependencies (requires uv)
uv sync

# Activate environment
source .venv/bin/activate  # macOS/Linux
```

### Running Notebooks

```bash
# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab
jupyter lab
```

### Testing Notebooks

```bash
# Execute notebook to verify it runs without errors
jupyter nbconvert --to notebook --execute lecture_N/notebook.ipynb --stdout > /dev/null
```

## Dependencies

From `pyproject.toml`:

- `jupyter` - Notebook environment
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - ML algorithms and datasets
- `ipykernel` - Jupyter kernel

## Course Architecture

The 12 lectures follow this progression:

**Lectures 1-4: Foundations**

- ML basics, word embeddings, tokenization, neural networks
- All examples use text/NLP data

**Lectures 5-6: Modern Architectures**

- Transformers, attention, pretraining, scaling laws

**Lectures 7-12: LLM Capabilities & Applications**

- Emergent abilities, alignment, RLHF, local models, prompting, RAG, agents

Each lecture builds on previous concepts and previews upcoming topics. The NLP/LLM focus is maintained throughout, even in foundational lectures.

## Notebook Development

### Pedagogical Principles

When creating or modifying lecture notebooks, follow these seven core principles:

#### 1. Structure with Clear Signposting

Begin each notebook with a short bullet list of objectives (3-5 items). Use clear section headers (`#`, `##`) as navigation landmarks. End with resources and a questions/contact section. This creates psychological safety—students know where they are and where they're going.

**Example structure:**

- Title and metadata (Bulgarian)
- Learning objectives (3-5 bullet points)
- Motivation section
- Topic sections with clear headers
- Summary and bridge to next lecture
- Resources and questions/contact

#### 2. Embrace Extreme Brevity

Keep markdown explanations to **2-8 lines maximum**. Break complex ideas into multiple cells rather than one long exposition. For code, write **1-5 line cells** that do exactly one thing. This forces clarity and makes the material scannable. Students can grasp each piece before moving forward, and they can easily re-run specific experiments.

#### 3. Follow the Four-Beat Pattern

For each concept, use this rhythm:

1. **Short markdown** explaining the idea
2. **Minimal code** demonstrating it
3. **Visual output** (graph, tree diagram, numbers)
4. **Brief markdown** interpreting what we see

This pattern repeats throughout, creating a predictable learning cadence. The repetition builds confidence.

#### 4. Show, Don't Just Tell

Every theoretical concept should have **runnable code within 1-2 cells**. Use simple text examples (short sentences, small vocabularies) when they make the concept clearer. Show intermediate steps with print statements for tokenization, embeddings, or predictions. Display actual outputs—not pseudocode or abstracted examples. Include both successful cases and instructive failures (overfitting, poor tokenization, attention collapse).

**Datasets to use:**

- Real data: IMDB reviews, AG News, SMS Spam, WikiText, small subsets of Common Crawl
- Synthetic data: constructed sentences, toy vocabularies, simple dialogue examples
- Include both "from scratch" implementations and library usage (Hugging Face, sklearn)

#### 5. Visualize Relentlessly

Create or use helper functions for common plots (loss curves, attention heatmaps, embedding projections). Put visualizations **immediately adjacent** to the concept—never defer them. Use matplotlib/seaborn for standard plots, but don't hesitate to use specialized tools (t-SNE for embeddings, heatmaps for attention). Color-code token types or classes. Label axes clearly with linguistic meaning.

**Visualization requirements:**

- Training/validation loss curves for models
- Attention weight heatmaps (token-to-token)
- Embedding visualizations (2D projections via PCA/t-SNE)
- Token probability distributions
- Perplexity or other language modeling metrics over time
- Always use clear labels and legends

#### 6. Balance Rigor with Accessibility

Include mathematical notation (LaTeX) for precision, but **always follow with plain language**. Show formulas, then demonstrate them with simple numerical examples (e.g., `softmax([2.0, 1.0, 0.1])` → token probabilities). Provide intuition before formalism. Use questions as headers: "Какво е attention механизмът?" rather than just "Attention".

**Mathematical content guidelines:**

- LaTeX for equations in markdown cells
- Include key derivations inline
- Connect math to code implementation immediately
- Show concrete numerical examples
- Explain in Bulgarian prose before formulas

#### 7. Build Progressive Complexity

Start each topic at the **most concrete level possible**. Use tiny vocabularies (5-10 words) or single sentences even when models handle thousands of tokens. Show the simple case working perfectly, then show edge cases and failures. Only then introduce solutions (subword tokenization, positional encodings, attention masking). This mirrors natural problem-solving.

**Progression pattern:**

- Simple case (small vocab, short sequences, clear patterns)
- Show it working
- Introduce edge cases and failures (OOV words, long sequences, ambiguity)
- Explain why it fails
- Present solutions
- Show solutions working

## Working with This Codebase

**Current state**: Lectures 1-4 are complete with notebooks. Lectures 5-7 have detailed plans but incomplete notebooks. Lectures 8-12 have plans only.

**Bilingual nature**: Expect Bulgarian text in markdown cells and plot titles. Code and variable names remain in English.

**Data loading**: All datasets loaded programmatically (no data/ directory). Uses scikit-learn built-in datasets and similar sources.

**Notebook execution**: Designed to run top-to-bottom sequentially. Each notebook is mostly self-contained.

**Reference flow**: When adding content, check:

1. `course_overview.md` for high-level positioning
2. `ml-terms-en-bg.md` for terminology consistency
3. Previous lecture notebooks for style/structure patterns
4. Lecture plans for detailed topic coverage

## Git Information

- Main branch: `master`
- Current state: Clean working directory
- Ignore pattern: Virtual env, checkpoints, Python artifacts (see .gitignore)

## External References

- Stanford CS336 lecture course (see notes.txt)
- Stanford CS224N, CS229
- Papers and textbooks (detailed in individual lecture plans)
