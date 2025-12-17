# Изпит: Вариант C

**Инструкции:** Решете всички 8 задачи. Показвайте изчисленията си.

---

## Задача 1: TF-IDF и Text Representation (12 точки)

Имате колекция от 5 документа. Думата "transformer" се среща:
- Документ 1: 8 пъти (документ с 200 думи)
- Документ 2: 3 пъти (документ с 150 думи)
- Документ 3: 0 пъти
- Документ 4: 0 пъти
- Документ 5: 1 път (документ с 100 думи)

**Задача:**
a) Изчислете TF за "transformer" в документ 1
b) Изчислете IDF за "transformer" в колекцията
c) Изчислете TF-IDF тежестта за документ 1
d) Защо IDF дава по-висока тежест на редки думи?

**Формули:**
- $TF = \frac{\text{брой на думата}}{\text{общ брой думи}}$
- $IDF = \log\frac{\text{общ брой документи}}{\text{брой документи с думата}}$

---

## Задача 2: Positional Encoding и Attention Mask (12 точки)

Разглеждате decoder-only transformer за генерация.

**Задача:**
a) Попълнете causal attention mask за последователност от 5 токена (1 = може да вижда, 0 = не може):

```
     t1  t2  t3  t4  t5
t1 [  ?   ?   ?   ?   ?  ]
t2 [  ?   ?   ?   ?   ?  ]
t3 [  ?   ?   ?   ?   ?  ]
t4 [  ?   ?   ?   ?   ?  ]
t5 [  ?   ?   ?   ?   ?  ]
```

b) Защо decoder използва causal mask, а encoder (BERT) не?
c) Как RoPE (Rotary Position Embeddings) кодира позиция?
d) Защо относителната позиция е по-полезна от абсолютната за дълги последователности?

---

## Задача 3: Feed-Forward Network и Residual (15 точки)

Transformer блок има:
- $d_{model} = 512$
- FFN expansion factor: 4x
- Activation: GELU

FFN формула: $FFN(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2$

С residual connection: $output = x + FFN(LayerNorm(x))$

**Задача:**
a) Каква е размерността на скрития слой в FFN?
b) Изчислете броя параметри за $W_1$ (без bias)
c) Изчислете броя параметри за $W_2$ (без bias)
d) Защо residual connections са критични за дълбоки мрежи?
e) Каква е разликата между Pre-Norm и Post-Norm? Кой се използва в модерните LLM?

---

## Задача 4: Emergence и Few-Shot Learning (12 точки)

Benchmark резултати за in-context learning:

| Модел | Параметри | 0-shot | 1-shot | 5-shot | 10-shot |
|-------|-----------|--------|--------|--------|---------|
| A     | 1B        | 28%    | 29%    | 30%    | 31%     |
| B     | 10B       | 35%    | 42%    | 51%    | 54%     |
| C     | 100B      | 48%    | 62%    | 74%    | 78%     |

**Задача:**
a) Кой модел показва най-силен emergence на few-shot learning?
b) За модел C, каква е ползата от добавяне на примери от 5-shot към 10-shot?
c) Защо малките модели (1B) почти не се подобряват с повече примери?
d) При какъв размер (~параметри) few-shot learning става надеждно?

---

## Задача 5: KV Cache и Memory (12 точки)

Модел с:
- 32 layers
- $d_{model} = 4096$
- 32 attention heads
- FP16 precision (2 bytes per value)

При генерация се съхранява KV cache за всички предишни токени.

**Опростена формула:** KV cache ≈ $2 \times layers \times d_{model} \times seq\_len \times 2$ bytes

**Задача:**
a) Изчислете KV cache размер за 2048 токена context
b) Изчислете KV cache размер за 8192 токена context
c) Ако GPU има 24GB VRAM и model weights са 14GB, колко токена context можете да използвате?
d) Как GQA (Grouped Query Attention) намалява KV cache размера?

---

## Задача 6: SFT vs RLHF (12 точки)

Процесът на alignment включва:
1. Pretrained model → SFT → RLHF model

**SFT данни:** 50,000 instruction-response примера
**RLHF данни:** 100,000 preference comparisons (A > B)

**Задача:**
a) Какъв е основният loss при SFT?
b) Как се обучава reward model за RLHF?
c) Защо SFT не е достатъчен за пълен alignment?
d) Дайте пример за поведение, което RLHF може да коригира, но SFT не може
e) Какво е Constitutional AI и как се различава от стандартен RLHF?

---

## Задача 7: Vector Search и Similarity (13 точки)

RAG система търси в база от 50,000 документа.

Query embedding: $q = [0.5, 0.8, 0.3]$

Три candidate документа:
- $d_1 = [0.6, 0.7, 0.4]$
- $d_2 = [-0.2, 0.9, 0.1]$
- $d_3 = [0.4, 0.8, 0.2]$

**Формула:** $\cos(q, d) = \frac{q \cdot d}{||q|| \cdot ||d||}$

**Задача:**
a) Изчислете dot product $q \cdot d_1$
b) Без да изчислявате пълен cosine similarity, кой документ е най-близък до query? (hint: сравнете dot products)
c) Защо cosine similarity е предпочитан пред Euclidean distance за embeddings?
d) Какво е Approximate Nearest Neighbor (ANN) и защо е необходим при 50K+ документа?

---

## Задача 8: Prompt Engineering и Output Control (12 точки)

Имате задача за извличане на информация от текст.

**Текст:** "Apple Inc. reported Q3 revenue of $81.8 billion, a 2% increase year-over-year. CEO Tim Cook announced new AI features for iPhone 16."

**Цели:**
- Извличане на: компания, приход, промяна, CEO, продукт
- Output във формат JSON

**Задача:**
a) Напишете zero-shot prompt за тази задача
b) Напишете few-shot prompt с 1 пример (измислете подобен текст и очакван output)
c) Кой подход ще даде по-надежден JSON output?
d) Какво е "output format control" и защо е важно за production системи?
e) Как бихте валидирали, че output-ът е валиден JSON?

---

# Скала за оценяване

| Точки | Оценка |
|-------|--------|
| 90-100 | Отличен (6) |
| 75-89 | Много добър (5) |
| 60-74 | Добър (4) |
| 50-59 | Среден (3) |
| < 50 | Слаб (2) |
