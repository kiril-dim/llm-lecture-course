# Кратки упражнения: Лекция 11

Следните упражнения са за самостоятелна работа по време на лекцията или веднага след нея. Очаквано време: 2-3 минути за упражнение.

---

## Упражнение 1: Категории халюцинации

Класифицирайте следните грешки на модела:

1. "Eiffel Tower е построена през 1920 г." (истина: 1889)
2. "Според Smith et al. (2019) в Journal of AI Research..." (цитатът не съществува)
3. "Слънцето обикаля около Земята"
4. "Python 4.0 беше пуснат миналия месец" (не е)

**Категории:**
- A) Фактическа грешка
- B) Фабрикувана цитация
- C) Невъзможно твърдение
- D) Остаряла/невярна актуална информация

---

## Упражнение 2: Защо LLM халюцинират?

LLM са обучени да предсказват най-вероятния следващ токен.

**Въпроси:**
1. Защо "най-вероятен текст" не означава "верен текст"?
2. Ако training data съдържа грешки, какво научава моделът?
3. Защо моделът звучи уверен дори когато греши?

---

## Упражнение 3: RAG Pipeline

Подредете стъпките в правилен ред:

- A) Генериране на отговор с LLM
- B) Търсене в vector database
- C) Chunking на документи
- D) Получаване на query от потребителя
- E) Embedding на query
- F) Форматиране на retrieved chunks в prompt

**Правилен ред:** ? → ? → ? → ? → ? → ?

---

## Упражнение 4: Semantic vs Keyword Search

Query: "Как да подобря производителността на Python код?"

**Document A:** "Python performance optimization includes profiling, caching, and using efficient data structures."

**Document B:** "Python е език за програмиране, създаден от Guido van Rossum."

**Въпроси:**
1. Кой документ ще върне keyword search за "Python производителност"?
2. Кой документ ще върне semantic search?
3. Защо semantic search е по-подходящ за RAG?

---

## Упражнение 5: Cosine Similarity

Имате три embedding вектора (опростени до 2D):

- Query: [0.8, 0.6]
- Doc A: [0.9, 0.5]
- Doc B: [-0.3, 0.9]

**Формула:** $\cos(\theta) = \frac{A \cdot B}{||A|| \cdot ||B||}$

**Задача:**
1. Кой документ е семантично по-близък до query?
2. Какво означава negative cosine similarity?

**Hint:** Dot product: Query·A = 0.72+0.30 = 1.02, Query·B = -0.24+0.54 = 0.30

---

## Упражнение 6: Chunking Strategies

Документ от 10,000 думи трябва да се раздели на chunks.

| Стратегия | Chunk size | Overlap |
|-----------|------------|---------|
| A | 100 думи | 0 |
| B | 500 думи | 50 думи |
| C | 500 думи | 100 думи |

**Въпроси:**
1. Колко chunks ще има при всяка стратегия?
2. Кога overlap е важен?
3. Защо много малки chunks са проблем за retrieval?

---

## Упражнение 7: Top-k Retrieval

При retrieval с k=5 получавате:

| Rank | Similarity | Relevant? |
|------|------------|-----------|
| 1 | 0.92 | Да |
| 2 | 0.85 | Да |
| 3 | 0.78 | Не |
| 4 | 0.71 | Да |
| 5 | 0.65 | Не |

**Задача:** Изчислете:
1. Precision@5 (релевантни / върнати)
2. Какво се случва ако намалим k на 3?
3. Защо включването на нерелевантни chunks е проблем?

---

## Упражнение 8: Vector Database Comparison

| База | Тип | Предимство |
|------|-----|------------|
| Chroma | In-memory/persistent | ? |
| Pinecone | Managed cloud | ? |
| pgvector | PostgreSQL extension | ? |

**Задача:** Съпоставете предимствата:
- A) Интегрира се с existing PostgreSQL инфраструктура
- B) Лесен setup, добър за прототипиране
- C) Scalability без инфраструктурни грижи

---

## Упражнение 9: Embedding Model Trade-offs

| Модел | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Бърз | Среден |
| text-embedding-3-small | 1536 | Среден | Добър |
| text-embedding-3-large | 3072 | Бавен | Най-добър |

**Въпроси:**
1. Защо повече dimensions = по-добро quality?
2. Какъв е trade-off при избор на по-голям модел?
3. За RAG върху 1M документа кой модел бихте избрали?

---

## Упражнение 10: Context Window Management

LLM има 8K token context window. Имате:
- System prompt: 200 tokens
- User query: 100 tokens
- Очакван отговор: 500 tokens
- Retrieved chunks: 5 chunks × 600 tokens = 3000 tokens

**Въпроси:**
1. Колко tokens остават за допълнителен context?
2. Какво става ако retrieved chunks са 10 × 600 = 6000 tokens?
3. Как бихте решили проблема с твърде много retrieved content?

---

## Упражнение 11: HyDE (Hypothetical Document Embeddings)

Стандартен RAG: embed query → search
HyDE: generate hypothetical answer → embed it → search

**Пример:**
- Query: "Каква е столицата на Австралия?"
- Hypothetical answer: "Столицата на Австралия е [grad], който е...'

**Въпроси:**
1. Защо hypothetical answer може да е по-добър за search?
2. Какъв е недостатъкът на HyDE?
3. Кога HyDE помага най-много?

---

## Упражнение 12: RAG Failure Modes

Диагностицирайте проблема:

**Сценарий 1:** RAG връща правилни документи, но LLM дава грешен отговор.

**Сценарий 2:** RAG връща нерелевантни документи, въпреки че правилните съществуват в базата.

**Сценарий 3:** RAG работи добре за кратки въпроси, но се проваля за дълги.

**Задача:** За всеки сценарий определете:
- Къде е проблемът (retrieval или generation)?
- Възможна причина
- Как бихте го debug-нали?

---

# Решения

## Решение 1
1. "Eiffel Tower 1920" → A) Фактическа грешка
2. "Smith et al. (2019)" → B) Фабрикувана цитация
3. "Слънцето обикаля Земята" → C) Невъзможно твърдение
4. "Python 4.0" → D) Остаряла/невярна актуална информация

## Решение 2
1. Training data съдържа много текст, който звучи правдоподобно, но е грешен. Моделът учи patterns, не истина.
2. Моделът възпроизвежда грешките - "garbage in, garbage out"
3. Confident tone е part of training data - експерти пишат уверено, моделът имитира стила

## Решение 3
Правилен ред: **C → D → E → B → F → A**
1. C) Chunking на документи (preprocessing)
2. D) Получаване на query от потребителя
3. E) Embedding на query
4. B) Търсене в vector database
5. F) Форматиране на retrieved chunks в prompt
6. A) Генериране на отговор с LLM

## Решение 4
1. Keyword search: Може да върне и двата или само B (зависи от exact match на "Python")
2. Semantic search: Document A (семантично близък до оптимизация на производителност)
3. Semantic search разбира meaning, не само keywords - "производителност" и "performance optimization" са свързани концепции

## Решение 5
1. Doc A е по-близък (по-висок dot product: 1.02 vs 0.30, и vectors point in similar direction)
2. Negative cosine similarity означава vectors сочат в противоположни посоки - документът е семантично несвързан или противоположен

## Решение 6
1. Брой chunks:
   - A: 10,000/100 = 100 chunks
   - B: ~22 chunks (10,000-50)/(500-50) ≈ 22
   - C: ~25 chunks (10,000-100)/(500-100) ≈ 25
2. Overlap е важен когато важна информация може да попадне на границата между chunks
3. Много малки chunks губят context - отделен chunk може да няма достатъчно информация за смислен retrieval

## Решение 7
1. Precision@5 = 3/5 = 60%
2. При k=3: Precision@3 = 2/3 = 67% (по-добър, но може да пропуснем relevant document #4)
3. Нерелевантни chunks "замърсяват" context-а, могат да объркат LLM или да изместят важна информация

## Решение 8
- Chroma → B) Лесен setup, добър за прототипиране
- Pinecone → C) Scalability без инфраструктурни грижи
- pgvector → A) Интегрира се с existing PostgreSQL инфраструктура

## Решение 9
1. Повече dimensions = повече "пространство" за представяне на семантични нюанси
2. Trade-off: По-бавен embedding, повече storage, по-бавен search
3. Зависи от use case: all-MiniLM за бързина и евтино storage, text-embedding-3-small за добър баланс

## Решение 10
1. Свободни tokens: 8000 - 200 - 100 - 500 - 3000 = 4200 tokens
2. При 6000 tokens за chunks: 8000 - 200 - 100 - 500 - 6000 = 1200 tokens - твърде малко margin
3. Решения: намалете k, summarize chunks, използвайте re-ranking за по-добър подбор, truncate дълги chunks

## Решение 11
1. Hypothetical answer е по-близък до документите в embedding space - query-то е въпрос, но документите съдържат отговори (statements)
2. Недостатък: Допълнителен LLM call = повече latency и cost
3. HyDE помага най-много при queries, които са много различни по форма от документите (въпроси vs твърдения)

## Решение 12
**Сценарий 1:** Проблем в generation
- Причина: LLM не следва context, hallucination въпреки correct retrieval
- Debug: Проверете prompt формата, опитайте по-explicit instructions

**Сценарий 2:** Проблем в retrieval
- Причина: Лош embedding model, неподходящ chunking, query/document mismatch
- Debug: Инспектирайте similarity scores, проверете какво се retrieve-ва

**Сценарий 3:** Проблем в retrieval (embedding на дълги queries)
- Причина: Дълги queries се embed-ват зле или се truncate
- Debug: Опитайте query reformulation, извлечете ключови части от query-то
