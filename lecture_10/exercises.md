# Кратки упражнения: Лекция 10

Следните упражнения са за самостоятелна работа по време на лекцията или веднага след нея. Очаквано време: 2-3 минути за упражнение.

---

## Упражнение 1: Zero-Shot vs Few-Shot CoT

Разгледайте три подхода за math word problem:

**Zero-shot:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans with 3 balls each.
How many tennis balls does he have now?
A:
```

**Zero-shot CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans with 3 balls each.
How many tennis balls does he have now?
A: Let's think step by step.
```

**Few-shot CoT:**
```
Q: Tim has 3 apples. He buys 2 bags with 4 apples each. How many apples?
A: Tim starts with 3 apples. He buys 2 bags × 4 apples = 8 apples. Total: 3 + 8 = 11.

Q: Roger has 5 tennis balls. He buys 2 more cans with 3 balls each.
How many tennis balls does he have now?
A:
```

**Въпроси:**
1. Кой подход ще даде най-точен отговор и защо?
2. Кой използва най-малко токени?
3. При какъв размер модел Zero-shot CoT започва да работи надеждно?

---

## Упражнение 2: Chain of Thought Ефективност

Резултати на GSM8K (math word problems) за различни модели:

| Модел | Standard | + CoT | Подобрение |
|-------|----------|-------|------------|
| GPT-3 6.7B | 6% | 9% | +3pp |
| GPT-3 175B | 18% | 57% | +39pp |
| PaLM 62B | 33% | 58% | +25pp |
| PaLM 540B | 56% | 74% | +18pp |

**Въпроси:**
1. За кой модел CoT дава най-голямо абсолютно подобрение?
2. Защо малките модели (6.7B) имат минимална полза от CoT?
3. Какво е "emergence threshold" за CoT?

---

## Упражнение 3: Self-Consistency

Self-consistency генерира N chain-of-thought отговора и взима majority vote.

**Пример:** Задача с 5 CoT samples:
- Sample 1: "... отговорът е 42"
- Sample 2: "... отговорът е 38"
- Sample 3: "... отговорът е 42"
- Sample 4: "... отговорът е 42"
- Sample 5: "... отговорът е 40"

**Въпроси:**
1. Какъв е финалният отговор?
2. Колко пъти повече inference струва self-consistency с N=5?
3. Ако базовата accuracy е 60%, каква accuracy очаквате с N=5?

**Hint:** При independent samples, majority от 5 е верен ако ≥3 са верни.

---

## Упражнение 4: Self-Consistency Cost-Benefit

Benchmark резултати със self-consistency:

| N (samples) | Accuracy | Cost (× baseline) |
|-------------|----------|-------------------|
| 1 | 60% | 1× |
| 3 | 68% | 3× |
| 5 | 71% | 5× |
| 10 | 74% | 10× |
| 20 | 76% | 20× |
| 40 | 77% | 40× |

**Въпроси:**
1. Къде е diminishing returns?
2. Колко струва 1 процентна точка подобрение при N=40 vs N=3?
3. При какъв N бихте спрели за production система?

---

## Упражнение 5: Tree of Thoughts

Tree of Thoughts търси в пространството от reasoning paths.

**Пример: Game of 24**
Числа: 4, 5, 6, 10
Цел: Използвай +, -, ×, ÷ за да получиш 24

**Въпроси:**
1. Защо линеен CoT може да се "закючи" в грешна посока?
2. Колко LLM calls прави ToT в сравнение с един CoT?
3. Защо ToT рядко се използва в production?

---

## Упражнение 6: Process vs Outcome Supervision

Две стратегии за training на reward model:

**Outcome supervision:** Reward само за финален верен отговор
**Process supervision:** Reward за всяка вярна стъпка

**Пример:**
```
Стъпка 1: 5 + 3 = 8 ✓
Стъпка 2: 8 × 2 = 15 ✗ (трябва да е 16)
Стъпка 3: 15 + 1 = 16 ✓
Финален отговор: 16 ✓
```

**Въпроси:**
1. Какъв reward дава outcome supervision?
2. Какъв reward дава process supervision?
3. Защо process supervision е по-надежден за training?

---

## Упражнение 7: Test-Time Compute Scaling

Reasoning модели (o1, R1) използват повече compute по време на inference.

**Данни за o1:**

| Thinking time | Accuracy (AIME) |
|---------------|-----------------|
| 1 секунда | 45% |
| 10 секунди | 62% |
| 60 секунди | 74% |
| 5 минути | 83% |

**Въпроси:**
1. Каква е връзката между thinking time и accuracy?
2. При какъв тип задачи това е полезно?
3. Защо "thinking longer" помага, ако weights не се променят?

---

## Упражнение 8: Reasoning Model Trade-offs

Сравнение на подходи:

| Подход | Latency | Cost | Accuracy (hard math) |
|--------|---------|------|---------------------|
| GPT-4 standard | ~2s | 1× | 65% |
| GPT-4 + CoT | ~4s | 1.5× | 78% |
| GPT-4 + Self-consistency (5) | ~10s | 7× | 82% |
| o1-mini | ~30s | 3× | 88% |
| o1 | ~2min | 15× | 94% |

**Въпроси:**
1. За "бърз отговор на прост въпрос", кой е най-добър?
2. За "критичен финансов анализ", кой е най-добър?
3. Кой подход е най-добър "bang for buck"?

---

## Упражнение 9: Prompt Engineering Techniques

Класифицирайте следните техники:

1. "You are an expert mathematician. Solve this problem:"
2. "Let's think step by step."
3. "Answer in JSON format: {\"result\": ...}"
4. "Example 1: ... Example 2: ... Now solve:"
5. ```Solve the following problem:
   ###
   [problem text]
   ###```

**Категории:**
- A) Role/persona
- B) Chain of thought trigger
- C) Output format control
- D) Few-shot examples
- E) Delimiter structure

---

## Упражнение 10: When to Use What

Съпоставете задачи с подходящи техники:

**Задачи:**
1. Извличане на имена от текст
2. Многостъпкова математика
3. Критичен код за production
4. Sentiment на 1000 коментара
5. Превод на изречение

**Техники:**
- A) Zero-shot с format instructions
- B) CoT
- C) Reasoning model (o1)
- D) Few-shot примери
- E) Standard prompt

---

## Упражнение 11: CoT Failure Modes

Кога Chain of Thought НЕ помага или вреди?

**Сценарии:**
1. "What is the capital of France?"
2. "Is 847 × 293 > 250000?"
3. "Write a haiku about autumn."
4. "What year did World War 2 end?"
5. "If all roses are flowers, and some flowers fade quickly, do some roses fade quickly?"

**Въпроси:**
1. За кои от тези CoT помага?
2. За кои CoT е безполезен overhead?
3. За кои CoT може да "навреди" (overthinking)?

---

## Упражнение 12: Evolution Timeline

Подредете по хронология:

- A) Tree of Thoughts (Yao et al.)
- B) GPT-3 Few-shot Learning (Brown et al.)
- C) DeepSeek R1
- D) Chain of Thought (Wei et al.)
- E) Self-Consistency (Wang et al.)
- F) Zero-shot CoT "Let's think step by step" (Kojima et al.)
- G) OpenAI o1

**Въпроси:**
1. Подредете A-G по година на публикуване
2. Колко години отнема от few-shot до reasoning models?
3. Защо discovery на CoT отнема 2 години след GPT-3?

---

# Решения

## Решение 1
1. Few-shot CoT - показва точния формат на reasoning, който очакваме
2. Zero-shot (най-малко токени, но и най-ненадежден)
3. ~100B+ параметъра - под този праг моделите не следват CoT инструкциите надеждно

## Решение 2
1. GPT-3 175B (+39pp) - най-голямо абсолютно подобрение
2. Малките модели нямат "emergence" на reasoning - CoT изисква способност да следват многостъпкова логика, която emerge при по-голям scale
3. ~50-100B параметъра - под този праг CoT не помага значително

## Решение 3
1. 42 (majority vote: 3 от 5)
2. 5× повече inference cost
3. ~73-75%. При 60% base accuracy, вероятността поне 3 от 5 да са верни е: P(3) + P(4) + P(5) ≈ 0.68

## Решение 4
1. След N=10 (само 3pp за удвояване на samples)
2. N=40: 1pp струва 20× baseline; N=3: 1pp струва ~0.4× baseline (8pp за 3×)
3. N=5 или N=10 - добър баланс между accuracy и cost

## Решение 5
1. Линеен CoT прави една последователност от стъпки - ако първата стъпка е грешна посока, няма как да backtrack
2. O(b^d) calls, където b е branching factor и d е depth - много повече от 1
3. Твърде скъпо (много API calls), твърде бавно, сложно за имплементация; reasoning models интернализират подобна логика

## Решение 6
1. Outcome: reward = 1 (финалният отговор 16 е верен)
2. Process: reward = 2/3 (2 верни стъпки от 3)
3. Process supervision открива "lucky" верни отговори с грешен reasoning - помага на модела да учи истински правилна логика, не просто да налучква

## Решение 7
1. Логаритмична (log-linear) - повече време = по-висока accuracy, но с diminishing returns
2. Сложни reasoning задачи - математика, coding, логически пъзели; НЕ за прости factual queries
3. Моделът генерира повече intermediate tokens (reasoning), което му позволява да "explore" различни подходи и да се self-correct

## Решение 8
1. GPT-4 standard - бърз и евтин за прости въпроси
2. o1 - когато accuracy е критична, cost е вторичен
3. GPT-4 + CoT - 78% accuracy при само 1.5× cost е отличен trade-off

## Решение 9
1. A - Role/persona
2. B - Chain of thought trigger
3. C - Output format control
4. D - Few-shot examples
5. E - Delimiter structure

## Решение 10
1. Извличане на имена → D (Few-shot examples показват формата)
2. Многостъпкова математика → B (CoT)
3. Критичен код → C (Reasoning model за максимална accuracy)
4. Sentiment на 1000 коментара → A (Zero-shot с format, бързо и евтино)
5. Превод → E (Standard prompt, добре trained task)

## Решение 11
1. CoT помага: 2 (математика), 5 (логика)
2. CoT безполезен: 1 (factual recall), 4 (factual recall)
3. CoT може да навреди: 3 (creative writing - overthinking убива spontaneity)

## Решение 12
1. Хронология:
   - B) GPT-3 Few-shot (2020)
   - D) Chain of Thought (2022)
   - F) Zero-shot CoT (2022)
   - E) Self-Consistency (2022)
   - A) Tree of Thoughts (2023)
   - G) OpenAI o1 (2024)
   - C) DeepSeek R1 (2025)

2. ~4-5 години (2020 → 2024/25)
3. Изследователите се фокусираха върху scaling и architecture промени; prompting изглеждаше "твърде просто" за да е важно; serendipitous discovery
