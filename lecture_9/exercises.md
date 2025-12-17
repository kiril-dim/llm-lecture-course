# Кратки упражнения: Лекция 9

Следните упражнения са за самостоятелна работа по време на лекцията или веднага след нея. Очаквано време: 2-3 минути за упражнение.

---

## Упражнение 1: Изчисляване на Memory за Weights

Модел с 7 милиарда параметъра (7B) съхранява weights в различни precision формати.

**Задача:** Изчислете размера на weights в GB за:
1. FP32 (32 бита на параметър)
2. FP16 (16 бита на параметър)
3. INT8 (8 бита на параметър)
4. INT4 (4 бита на параметър)

**Hint:** 1 GB = 8 милиарда бита

---

## Упражнение 2: VRAM Изисквания

Имате GPU с 8GB VRAM. Искате да заредите модел локално.

| Модел | Параметри | INT4 размер |
|-------|-----------|-------------|
| Phi-3 | 3.8B | ~2.3 GB |
| Mistral | 7B | ~4.2 GB |
| Llama 3 | 8B | ~4.8 GB |
| Llama 3 | 70B | ~40 GB |

**Въпроси:**
1. Кои модели се побират в 8GB VRAM?
2. Защо реалният VRAM е по-висок от размера на weights?
3. Какво е KV cache и защо заема допълнителна памет?

---

## Упражнение 3: Quantization Trade-offs

Сравнение на Llama 3 8B при различни quantization нива:

| Quantization | Размер | Perplexity | Относителна грешка |
|--------------|--------|------------|-------------------|
| FP16 | 16 GB | 6.14 | baseline |
| INT8 | 8 GB | 6.18 | +0.7% |
| INT4 (Q4_K_M) | 4.8 GB | 6.35 | +3.4% |
| INT3 | 3.2 GB | 7.12 | +16% |
| INT2 | 2.1 GB | 12.4 | +102% |

**Въпроси:**
1. При кое ниво quantization загубата на качество става значителна?
2. Какъв е trade-off между INT8 и INT4?
3. Защо INT2 е почти неизползваем за повечето задачи?

---

## Упражнение 4: Memory Bandwidth Bottleneck

LLM inference е memory-bound, не compute-bound.

**Данни:**
- Модел: 7B параметъра, INT4 (~4GB weights)
- GPU: RTX 3060 (192 GB/s bandwidth)
- За генериране на 1 токен трябва да се прочетат всички weights

**Задача:**
1. Колко време отнема да се прочетат 4GB при 192 GB/s?
2. Какъв е теоретичният максимум tokens/second?
3. Защо реалните резултати са по-ниски?

---

## Упражнение 5: Hardware Comparison

Сравнете три setup-а за локален LLM:

| Hardware | RAM/VRAM | Bandwidth | Цена |
|----------|----------|-----------|------|
| M2 MacBook Air | 16GB unified | 100 GB/s | ~$1200 |
| RTX 4060 Ti 16GB | 16GB VRAM | 288 GB/s | ~$450 (GPU) |
| RTX 4090 | 24GB VRAM | 1008 GB/s | ~$1600 (GPU) |

**Въпроси:**
1. Кой setup е най-бърз (tokens/second)?
2. Кой е най-добър за 70B модел и защо?
3. Какво е предимството на Apple Silicon unified memory?

---

## Упражнение 6: Model Size vs Use Case

Съпоставете размерите на модели с подходящи use cases:

**Модели:**
- A) 1-3B параметъра
- B) 7-8B параметъра
- C) 13-34B параметъра
- D) 70B+ параметъра

**Use cases:**
1. Code completion в IDE (бърз отговор критичен)
2. Customer support chatbot
3. Генериране на embeddings за RAG
4. Сложен code review с reasoning
5. Прост sentiment analysis
6. Превод на документи

---

## Упражнение 7: KV Cache Calculation

KV cache съхранява attention keys и values за всеки генериран токен.

**Формула:** KV cache size = 2 × layers × d_model × n_heads × sequence_length × bytes_per_value

**Данни за Llama 3 8B:**
- 32 layers
- d_model = 4096
- n_heads = 32
- FP16 (2 bytes per value)

**Задача:** Изчислете KV cache размер за:
1. 512 токена context
2. 4096 токена context
3. Какво се случва при много дълъг context?

---

## Упражнение 8: Local vs Cloud Cost

Сравнение на разходите за inference:

**Cloud (GPT-4 API):**
- Input: $0.03 / 1K tokens
- Output: $0.06 / 1K tokens

**Local (RTX 4090 + Llama 3 70B):**
- Hardware: $1600 (GPU) + $1500 (система) = $3100
- Електричество: $0.50/час при пълно натоварване

**Сценарий:** 1 милион output токена на ден

**Задача:**
1. Каква е дневната цена за cloud?
2. За колко дни се изплаща локалният setup?
3. При какъв обем cloud е по-изгоден?

---

## Упражнение 9: Deployment Tools

Съпоставете инструментите с подходящите сценарии:

**Инструменти:**
- A) llama.cpp
- B) Ollama
- C) vLLM

**Сценарии:**
1. Production server с много concurrent requests
2. Бърз setup за експериментиране на лаптоп
3. Максимална гъвкавост и контрол над inference
4. Docker-like модел management
5. PagedAttention за ефективен batch inference

---

## Упражнение 10: Quantization Methods

GPTQ, AWQ и GGUF са популярни quantization методи.

| Метод | Предимства | Недостатъци |
|-------|------------|-------------|
| GPTQ | Висока компресия, GPU-focused | По-бавна quantization |
| AWQ | Запазва важни weights | Само GPU |
| GGUF | CPU + GPU, flexible | Малко по-голям размер |

**Въпроси:**
1. Кой метод е най-подходящ за MacBook с Apple Silicon?
2. Кой е най-добър за production GPU сървър?
3. Защо AWQ запазва "важните" weights и как ги определя?

---

## Упражнение 11: Speculative Decoding

Speculative decoding използва малък "draft" модел за ускоряване.

**Идея:**
1. Draft модел (напр. 1B) генерира N токена бързо
2. Target модел (напр. 70B) верифицира всички наведнъж
3. Приемат се съвпадащите, останалите се регенерират

**Въпроси:**
1. Защо верификацията на N токена е по-бърза от генериране на N токена?
2. При какъв acceptance rate има смисъл speculative decoding?
3. Какво се случва ако draft модел е твърде различен от target?

---

## Упражнение 12: Hybrid Deployment

Компания иска hybrid setup (локален + cloud):

**Изисквания:**
- 80% от queries са прости (FAQ, рутинни въпроси)
- 15% изискват среден reasoning
- 5% са сложни, изискват топ модел

**Опции:**
- Локален 7B модел: $0.001/query
- Cloud GPT-4: $0.05/query
- Cloud GPT-4-turbo: $0.02/query

**Задача:** При 10,000 queries/ден:
1. Колко струва all-cloud (само GPT-4)?
2. Колко струва hybrid (7B за прости, GPT-4 за сложни)?
3. Каква е спестената сума на месец?

---

# Решения

## Решение 1
1. FP32: 7B × 32 бита = 224B бита = 28 GB
2. FP16: 7B × 16 бита = 112B бита = 14 GB
3. INT8: 7B × 8 бита = 56B бита = 7 GB
4. INT4: 7B × 4 бита = 28B бита = 3.5 GB

## Решение 2
1. Phi-3, Mistral и Llama 3 8B се побират
2. KV cache, activations и overhead на runtime заемат допълнителна памет
3. KV cache съхранява intermediate representations за attention - расте линейно с дължината на контекста

## Решение 3
1. При INT3 и по-ниско (+16% грешка е значителна)
2. INT4 е 1.7× по-малък с минимална загуба (+3.4%); INT8 има почти нулева загуба, но е 2× по-голям
3. При 2 бита информацията е твърде компресирана - невъзможно е да се възстановят детайлите на weights

## Решение 4
1. 4GB / 192 GB/s = 0.021 секунди = 21ms
2. 1000ms / 21ms ≈ 48 tokens/second теоретичен максимум
3. KV cache reads, CPU overhead, memory controller inefficiency, и реален bandwidth < максимален

## Решение 5
1. RTX 4090 (3.5× по-висок bandwidth от 4060 Ti)
2. Никой от тях директно - 70B INT4 е ~40GB. M2 и 4060 Ti нямат достатъчно памет. 4090 може с CPU offloading, но бавно
3. Unified memory позволява лесен достъп от CPU и GPU без копиране; може да използва повече от VRAM чрез swap

## Решение 6
1. Code completion → A (1-3B) - бързина е критична
2. Customer support chatbot → B (7-8B) - добър баланс
3. Embeddings за RAG → A (1-3B) - не изисква голям модел
4. Сложен code review → D (70B+) - изисква силен reasoning
5. Sentiment analysis → A (1-3B) - проста задача
6. Превод → B или C (7-34B) - зависи от езиковите двойки

## Решение 7
**Формула:** 2 × 32 × 4096 × 32 × seq_len × 2 bytes = 16,777,216 × seq_len bytes

1. 512 токена: 16.7M × 512 = 8.6 GB
2. 4096 токена: 16.7M × 4096 = 68.7 GB

*Корекция: По-точната формула дава по-малки стойности в практиката поради head dimension, но принципът е същият - KV cache расте линейно с context length и може да надмине размера на weights.*

3. При много дълъг context, KV cache може да надмине VRAM лимита - затова съществуват техники като sliding window attention

## Решение 8
1. Cloud: 1M × $0.06/1K = $60/ден
2. Local hardware $3100, електричество ~$12/ден ако работи непрекъснато
3. Break-even: $3100 / ($60 - $12) = ~65 дни

При по-нисък обем (под ~50K tokens/ден) cloud е по-изгоден

## Решение 9
1. vLLM (C) - оптимизиран за production throughput
2. Ollama (B) - лесен setup, добър CLI
3. llama.cpp (A) - пълен контрол, C++ библиотека
4. Ollama (B) - `ollama pull`, `ollama run`
5. vLLM (C) - PagedAttention е ключова feature

## Решение 10
1. GGUF - поддържа Metal backend за Apple Silicon, работи с CPU+GPU
2. AWQ или GPTQ - оптимизирани за GPU inference
3. AWQ анализира activation magnitudes - weights които водят до големи activations са по-важни и се quantize по-внимателно

## Решение 11
1. Верификацията е един forward pass за N токена (паралелизира се); генерирането е N последователни forward passes
2. Acceptance rate > 70-80% - иначе overhead от draft model и rejected tokens не си струва
3. Нисък acceptance rate - много rejection и re-generation, по-бавно от директно генериране

## Решение 12
1. All-cloud GPT-4: 10,000 × $0.05 = $500/ден
2. Hybrid:
   - 8,000 прости × $0.001 = $8
   - 1,500 средни × $0.02 = $30
   - 500 сложни × $0.05 = $25
   - Общо: $63/ден
3. Спестено: ($500 - $63) × 30 = $13,110/месец
