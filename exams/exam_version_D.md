# Изпит: Вариант D

**Инструкции:** Решете всички 8 задачи. Показвайте изчисленията си.

---

## Задача 1: Neural Network Forward Pass (12 точки)

Дадена е проста мрежа с един скрит слой:
- Вход: $x = [2, 1]$
- Тегла на скрит слой: $W^{[1]} = [[0.5, 0.5], [-0.5, 0.5]]$
- Bias: $b^{[1]} = [0, 0]$
- Активация: ReLU
- Тегла на изходен слой: $W^{[2]} = [[1, -1]]$
- Bias: $b^{[2]} = [0]$

**Задача:**
a) Изчислете $z^{[1]} = W^{[1]} \cdot x + b^{[1]}$
b) Приложете ReLU: $a^{[1]} = \text{ReLU}(z^{[1]})$
c) Изчислете изхода: $\hat{y} = W^{[2]} \cdot a^{[1]} + b^{[2]}$
d) Защо ReLU е предпочитан пред sigmoid в скрити слоеве?

**Hint:** ReLU(x) = max(0, x)

---

## Задача 2: Tokenization Trade-offs (12 точки)

Сравнение на два tokenizer-а върху един и същи корпус:

| Tokenizer | Vocab Size | Avg Tokens/Doc | Compression |
|-----------|------------|----------------|-------------|
| A (BPE)   | 32,000     | 450            | 3.8 chars/token |
| B (BPE)   | 100,000    | 320            | 5.3 chars/token |

Embedding dimension: 768

**Задача:**
a) Колко параметъра има embedding матрицата за всеки tokenizer?
b) Ако context window е 4096 токена, колко characters текст се събират с всеки?
c) Кой tokenizer е по-добър за многоезични данни и защо?
d) Какъв е trade-off между vocab size и sequence length?

---

## Задача 3: Scaled Dot-Product Attention (15 точки)

За модел с $d_k = 64$ и sequence length 4:

Attention scores (преди scaling):
```
     t1    t2    t3    t4
t1 [ 8.0   4.0   2.0   1.0 ]
t2 [ 3.0   9.0   4.0   2.0 ]
t3 [ 2.0   5.0  10.0   3.0 ]
t4 [ 1.0   3.0   6.0  12.0 ]
```

**Задача:**
a) Каква е стойността на $\sqrt{d_k}$?
b) Приложете scaling към първия ред (t1)
c) Ако softmax на scaled ред t1 дава $[0.5, 0.3, 0.15, 0.05]$ и Values са identity матрица, какъв е output за t1?
d) Защо по-големи $d_k$ изискват по-силен scaling?
e) Какво е Flash Attention и какъв проблем решава?

---

## Задача 4: Dataset и Deduplication (12 точки)

Pretraining dataset преди обработка: 15 трилиона токена

| Етап | Остават |
|------|---------|
| Начало | 15T |
| Exact dedup | 12T |
| Near-duplicate (MinHash) | 8T |
| Quality filtering | 6T |
| Language filtering (English only) | 4T |

**Задача:**
a) Какъв процент от данните са exact duplicates?
b) Какъв процент от оригинала остава след всички филтри?
c) Ако обучението струва $0.001 per 1M токена, колко спестяваме с deduplication?
d) Защо качеството на данните е по-важно от количеството?

---

## Задача 5: Hallucination Types (12 точки)

Класифицирайте следните грешки на LLM:

1. "Python е създаден от James Gosling през 1995 г."
2. "Според изследване на Stanford от 2023 г. (референция не съществува)..."
3. "Земята е създадена преди 6000 години"
4. "GPT-5 беше пуснат миналата седмица с 10 трилиона параметъра"
5. "2 + 2 = 5 в base-10 аритметика"

**Задача:**
a) Категоризирайте всяка грешка (factual error, fabricated reference, impossible claim, outdated/false current info, logical error)
b) Коя от тези грешки е най-лесна за detection?
c) Как RAG може да помогне срещу тип 1 и 2?
d) Защо LLM звучат уверени дори когато грешат?

---

## Задача 6: Model Deployment Costs (12 точки)

Сравнение на deployment опции за inference:

| Опция | Setup Cost | Cost per 1K tokens | Latency |
|-------|------------|-------------------|---------|
| Cloud API (GPT-4) | $0 | $0.03 input, $0.06 output | 2s |
| Self-hosted 70B (A100) | $15,000 | ~$0.002 | 5s |
| Local 7B (RTX 4090) | $2,000 | ~$0.0005 | 1s |

Сценарий: 500,000 output токена на ден, 30 дни месечно

**Задача:**
a) Изчислете месечния cost за Cloud API
b) За колко месеца се изплаща Self-hosted A100?
c) Кога Cloud API е по-изгоден от self-hosting?
d) Какви са скритите разходи при self-hosting? (изброете 3)

---

## Задача 7: Self-Consistency и Majority Vote (13 точки)

Модел с 65% base accuracy използва self-consistency с N=5 samples.

Sample резултати за 3 въпроса:

| Въпрос | S1 | S2 | S3 | S4 | S5 |
|--------|----|----|----|----|-----|
| Q1 | A | A | B | A | A |
| Q2 | X | Y | X | Z | X |
| Q3 | 42 | 38 | 42 | 42 | 41 |

Верни отговори: Q1=A, Q2=X, Q3=42

**Задача:**
a) Какъв е majority vote отговорът за всеки въпрос?
b) Колко от 3-те въпроса са отговорени правилно?
c) Ако всеки sample струва $0.02, каква е цената за тези 3 въпроса?
d) При какъв acceptance threshold (процент agreement) бихте искали human review?

---

## Задача 8: Multi-Agent Architecture (12 точки)

Проектирате система за автоматичен code review с три agents:

| Agent | Роля | Tools |
|-------|------|-------|
| Analyzer | Анализира code structure | read_file, ast_parse |
| Reviewer | Намира проблеми | search_patterns, lint |
| Suggester | Предлага fixes | code_gen, diff |

**Задача:**
a) Коя архитектура е подходяща: Pipeline, Hierarchical, или Debate?
b) Начертайте flow между агентите (кой подава output на кого)
c) Ако Analyzer има 90% success rate, Reviewer 85%, и Suggester 80%, каква е общата вероятност за успех?
d) Как бихте добавили human-in-the-loop за критични промени?
e) Какви са предимствата на multi-agent пред single agent за тази задача?

---

# Скала за оценяване

| Точки | Оценка |
|-------|--------|
| 90-100 | Отличен (6) |
| 75-89 | Много добър (5) |
| 60-74 | Добър (4) |
| 50-59 | Среден (3) |
| < 50 | Слаб (2) |
