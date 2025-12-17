# Кратки упражнения: Лекция 8

Следните упражнения са за самостоятелна работа по време на лекцията или веднага след нея. Очаквано време: 2-3 минути за упражнение.

---

## Упражнение 1: Base Model vs Aligned Model

Разгледайте два отговора на "Как да хакна WiFi мрежата на съседа?"

**Base model:** "За да хакнеш WiFi мрежа, първо трябва да..."

**Aligned model:** "Не мога да помогна с неоторизиран достъп до чужди мрежи. Ако имате проблем с вашата собствена мрежа, мога да помогна с..."

**Въпроси:**
1. Кой от трите H (Helpful, Harmless, Honest) е най-релевантен тук?
2. Base model-ът "вреден" ли е, или просто продължава вероятния текст?
3. Защо alignment е необходим, ако base model е технически по-"helpful"?

---

## Упражнение 2: SFT Data Requirements

SFT dataset съдържа instruction-response двойки. Типични размери:

| Dataset | Примери |
|---------|---------|
| Alpaca | 52,000 |
| Dolly | 15,000 |
| OpenAssistant | 161,000 |

Pretraining използва ~1 трилион токена.

**Въпроси:**
1. SFT данните колко пъти по-малко са от pretraining?
2. Защо толкова малко данни са достатъчни за SFT?
3. Какво "учи" модела при SFT - ново знание или формат?

---

## Упражнение 3: Reward Model Training

Имате 3 отговора на един prompt, ранкирани от human annotator:

1. Отговор A (най-добър)
2. Отговор B (среден)
3. Отговор C (най-лош)

**Задача:** Колко comparison pairs се генерират?

**Въпроси:**
1. Изброете всички pairs (winner > loser)
2. Ако имате 5 отговора, колко pairs?
3. Защо comparisons са по-лесни за annotators от absolute scores?

**Hint:** Брой pairs = $\binom{n}{2} = \frac{n(n-1)}{2}$

---

## Упражнение 4: Bradley-Terry Loss

Bradley-Terry моделът за preference learning:

$$P(A > B) = \sigma(r_A - r_B) = \frac{1}{1 + e^{-(r_A - r_B)}}$$

където $r$ е reward score.

**Задача:** Ако $r_A = 2.0$ и $r_B = 0.5$:
1. Изчислете $r_A - r_B$
2. Изчислете $P(A > B)$

**Hints:**
- $e^{-1.5} \approx 0.223$
- $\sigma(1.5) = \frac{1}{1 + 0.223} \approx 0.82$

**Въпрос:** Какво означава $P(A > B) = 0.82$?

---

## Упражнение 5: KL Penalty в RLHF

RLHF objective включва KL penalty:

$$\text{maximize } \mathbb{E}[r(x, y)] - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

където $\beta$ контролира силата на penalty.

**Въпроси:**
1. Какво се случва ако $\beta = 0$ (без KL penalty)?
2. Какво се случва ако $\beta$ е много голямо?
3. Защо KL penalty е критичен за RLHF?

---

## Упражнение 6: Reward Hacking

Reward model е обучен да предпочита "helpful" отговори. Модел открива, че може да получи висок reward чрез:

- Много дълги отговори (повече = по-helpful?)
- Повторение на въпроса в отговора
- Използване на уверен тон независимо от accuracy

**Въпроси:**
1. Кое от тези е пример за Goodhart's Law?
2. Как KL penalty помага срещу reward hacking?
3. Дайте пример за reward hack, който може да е вреден.

---

## Упражнение 7: SFT vs RLHF

Сравнете двата етапа:

| Аспект | SFT | RLHF |
|--------|-----|------|
| Данни | ? | ? |
| Loss | ? | ? |
| Какво учи | ? | ? |

**Задача:** Попълнете таблицата

**Въпрос:** Защо RLHF е необходим след SFT?

---

## Упражнение 8: Constitutional AI

CAI използва "constitution" - списък с принципи. Пример:

```
Principle: "Avoid responses that are harmful or unethical"
```

**Self-critique процес:**
1. Model генерира отговор
2. Model критикува собствения си отговор спрямо принципа
3. Model ревизира отговора

**Въпроси:**
1. Защо self-critique е по-scalable от human feedback?
2. Какъв е рискът ако "constitution" е непълна?
3. CAI замества ли напълно човешки feedback?

---

## Упражнение 9: Inter-Annotator Agreement

Трима annotators ранкират 100 response pairs. Agreement:

| Pair | A1 | A2 | A3 |
|------|----|----|----|
| 1 | A>B | A>B | A>B |
| 2 | A>B | B>A | A>B |
| 3 | B>A | A>B | B>A |
| ... | ... | ... | ... |

Резултати: 70% unanimous agreement, 30% split.

**Въпроси:**
1. Какво означава ниско agreement за качеството на reward model?
2. Как обикновено се решават disagreements?
3. Възможно ли е да има "правилен" отговор при субективни preferences?

---

## Упражнение 10: Alignment Tax

Benchmark резултати преди и след alignment:

| Benchmark | Base Model | Aligned Model |
|-----------|------------|---------------|
| MMLU | 72% | 70% |
| GSM8K | 65% | 67% |
| TruthfulQA | 31% | 52% |
| HumanEval | 48% | 45% |

**Въпроси:**
1. На кои benchmarks alignment "вреди"?
2. На кои benchmarks alignment помага?
3. Има ли реален trade-off между capability и safety?

---

## Упражнение 11: Red Teaming Categories

Класифицирайте следните атаки:

1. "Ignore previous instructions and..."
2. "You are DAN (Do Anything Now)..."
3. "Translate this harmful text to French"
4. "Write a story where a character explains how to..."

**Категории:**
- A) Instruction injection
- B) Persona/roleplay jailbreak
- C) Language/encoding bypass
- D) Fiction framing

**Задача:** Съпоставете атаките с категориите

---

## Упражнение 12: RLHF Training Dynamics

При RLHF training наблюдавате:

| Epoch | Reward | KL | Quality (human eval) |
|-------|--------|----|---------------------|
| 0 | 0.5 | 0 | 60% |
| 10 | 1.2 | 2.1 | 75% |
| 20 | 1.8 | 5.3 | 78% |
| 30 | 2.5 | 12.4 | 72% |
| 40 | 3.2 | 25.1 | 65% |

**Въпроси:**
1. В коя точка reward hacking вероятно започва?
2. Какъв е оптималният stopping point?
3. Защо human eval намалява въпреки растящия reward?

---

# Решения

## Решение 1
1. Harmless - предотвратяване на вредно действие
2. Base model не е "вреден" по намерение - просто продължава най-вероятния текст от training data
3. "Helpful" в тесен смисъл (отговаря на въпроса) не е достатъчно - трябва да се балансират и Harmless и Honest

## Решение 2
1. Pretraining: ~1T токена, SFT: ~50K примера ×100 токена = 5M токена → разлика ~200,000×
2. SFT не учи ново знание - само "активира" съществуващите способности в правилния формат
3. Формат (instruction → response), не ново знание

## Решение 3
1. Pairs: (A>B), (A>C), (B>C) = 3 pairs
2. За 5 отговора: $\binom{5}{2} = \frac{5 \times 4}{2} = 10$ pairs
3. Comparisons са по-лесни защото не изискват absolute scale - хората интуитивно знаят кое е "по-добро", но не могат консистентно да дават числови оценки

## Решение 4
1. $r_A - r_B = 2.0 - 0.5 = 1.5$
2. $P(A > B) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.223} \approx 0.82$

Това означава, че моделът предсказва 82% вероятност отговор A да е по-добър от B.

## Решение 5
1. $\beta = 0$: Моделът оптимизира само reward → reward hacking, degenerate outputs
2. $\beta$ голямо: Моделът почти не се променя от reference → няма learning
3. KL penalty предотвратява прекомерно отдалечаване от pretrained model, запазвайки fluency и общи способности

## Решение 6
1. Всички три са примери за Goodhart's Law ("When a measure becomes a target, it ceases to be a good measure")
2. KL penalty ограничава колко много може да се промени поведението, предотвратявайки екстремни exploits
3. Пример: Модел дава много уверени отговори дори когато е грешен - "вреден" защото води до overreliance

## Решение 7
| Аспект | SFT | RLHF |
|--------|-----|------|
| Данни | Instruction-response pairs | Preferences (A vs B comparisons) |
| Loss | Cross-entropy (next token) | PPO reward + KL penalty |
| Какво учи | Format/style на отговори | Preferences between отговори |

RLHF е необходим защото SFT не разграничава качеството между "правилно форматирани" отговори - само учи формата.

## Решение 8
1. AI може да генерира много примери бързо и евтино; human feedback е скъп и бавен
2. Непълна constitution → модел може да нарушава неспоменати принципи или да намери loopholes
3. Не - CAI обикновено се комбинира с human feedback за финална валидация

## Решение 9
1. Ниско agreement означава шумни labels → reward model учи шумен сигнал → по-лошо alignment
2. Majority vote или weighted by annotator quality
3. Не винаги - много preferences са субективни. Целта е да се align с "средно човешко предпочитание"

## Решение 10
1. MMLU (-2pp), HumanEval (-3pp) - alignment "вреди" на raw capability benchmarks
2. TruthfulQA (+21pp!) - alignment помага значително за honesty
3. Има trade-off, но е малък. TruthfulQA показва, че alignment може да подобри важни аспекти значително

## Решение 11
1. "Ignore previous instructions..." → A) Instruction injection
2. "You are DAN..." → B) Persona/roleplay jailbreak
3. "Translate this harmful text..." → C) Language/encoding bypass
4. "Write a story where a character..." → D) Fiction framing

## Решение 12
1. Reward hacking вероятно започва около epoch 20-30 (KL скача рязко, human eval започва да пада)
2. Оптимален stopping point: epoch 20 (най-висок human eval: 78%)
3. Reward model не перфектно измерва quality - моделът се научава да "exploit"-ва reward model, не да бъде реално по-добър
