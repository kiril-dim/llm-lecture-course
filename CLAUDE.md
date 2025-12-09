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

### Structure Pattern
1. Title and metadata (Bulgarian)
2. Motivation section
3. Topic sections (theory + code + visualizations)
4. Summary and bridge to next lecture

### Code Examples
- Use real datasets: IMDB reviews, AG News, SMS Spam
- Include both "from scratch" implementations and library usage
- Add extensive visualizations (training curves, confusion matrices, etc.)
- Show failure cases and limitations

### Mathematical Content
- LaTeX for equations in markdown cells
- Include key derivations inline
- Connect math to code implementation

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
