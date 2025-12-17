# Кратки упражнения: Лекция 12

Следните упражнения са за самостоятелна работа по време на лекцията или веднага след нея. Очаквано време: 2-3 минути за упражнение.

---

## Упражнение 1: LLM vs Agent

Класифицирайте задачите - кои изискват agent (tool use) и кои могат да се решат с чист LLM:

1. "Напиши имейл до клиент"
2. "Какво е времето в София днес?"
3. "Изчисли 847 × 293"
4. "Обобщи този документ"
5. "Резервирай маса в ресторант за утре"
6. "Обясни теорията на относителността"

**Категории:**
- A) Чист LLM (само генерация)
- B) Agent (изисква tools)

---

## Упражнение 2: Agent Loop

Подредете стъпките в agent loop:

- A) Изпълнение на action
- B) Получаване на observation
- C) Reasoning (мислене)
- D) Избор на tool/action
- E) Получаване на user input

**Правилен ред за един цикъл:** ? → ? → ? → ? → ?

---

## Упражнение 3: Tool Definition

Разгледайте тази tool definition:

```json
{
  "name": "search_web",
  "description": "Search the web for current information",
  "parameters": {
    "query": {
      "type": "string",
      "description": "The search query"
    },
    "num_results": {
      "type": "integer",
      "default": 5
    }
  }
}
```

**Въпроси:**
1. Защо `description` е критично важен?
2. Какво се случва ако description е неясен?
3. Защо има default стойност за `num_results`?

---

## Упражнение 4: ReAct Pattern

Разгледайте ReAct trace:

```
Question: Колко е населението на столицата на Франция?

Thought: Трябва да намеря столицата на Франция, после населението ѝ.
Action: search("столица на Франция")
Observation: Столицата на Франция е Париж.

Thought: Сега знам че е Париж. Трябва да намеря населението.
Action: search("население на Париж")
Observation: Населението на Париж е около 2.1 милиона.

Thought: Имам отговора.
Action: finish("Населението на столицата на Франция (Париж) е около 2.1 милиона.")
```

**Въпроси:**
1. Защо моделът не търси директно "население на столицата на Франция"?
2. Каква е ролята на Thought стъпките?
3. Как Observation влияе на следващия Thought?

---

## Упражнение 5: Tool Selection

Agent има достъп до 4 tools:

| Tool | Описание |
|------|----------|
| calculator | Математически изчисления |
| search | Търсене в интернет |
| code_exec | Изпълнение на Python код |
| file_read | Четене на локален файл |

**Задача:** Кой tool би избрал agent за:
1. "Колко е 15% от 847?"
2. "Какви са новините днес?"
3. "Анализирай данните в sales.csv"
4. "Напиши и тествай функция за сортиране"

---

## Упражнение 6: Agent Memory Types

| Memory тип | Характеристика |
|------------|----------------|
| Short-term | ? |
| Long-term (episodic) | ? |
| Long-term (semantic) | ? |

**Задача:** Съпоставете:
- A) Запомня факти и знания между сесии
- B) Текущият conversation history
- C) Запомня минали взаимодействия и техните резултати

---

## Упражнение 7: Planning Decomposition

User request: "Създай презентация за climate change с данни и графики"

**Задача:** Разбийте на subtasks:
1. ?
2. ?
3. ?
4. ?

**Въпрос:** Защо директното изпълнение без план е проблематично?

---

## Упражнение 8: Error Accumulation

Agent изпълнява 5-стъпков план. Вероятност за успех на всяка стъпка: 90%.

**Задача:**
1. Каква е вероятността целият план да успее?
2. При 10 стъпки?
3. Защо това е фундаментален проблем за agents?

**Формула:** $P_{total} = P_1 \times P_2 \times ... \times P_n$

---

## Упражнение 9: Reflexion Pattern

Agent прави грешка:

```
Task: Намери имейла на CEO на Anthropic
Action: search("Anthropic CEO email")
Observation: Няма публично достъпен имейл.
Result: FAILED
```

**Reflexion:**
```
Reflection: Търсенето на директен имейл не работи.
Learned: Трябва да търся официална contact страница или LinkedIn.
New strategy: search("Anthropic contact page") или search("Dario Amodei LinkedIn")
```

**Въпроси:**
1. Как reflection помага при следващ опит?
2. Къде се съхранява learned информацията?
3. Може ли reflection да доведе до грешни изводи?

---

## Упражнение 10: Multi-Agent Architectures

| Архитектура | Описание |
|-------------|----------|
| Hierarchical | Manager agent делегира на specialized agents |
| Debate | Agents спорят и достигат консенсус |
| Pipeline | Output от един agent е input за следващия |

**Задача:** Коя архитектура е най-подходяща за:
1. Code review (един agent пише, друг критикува)
2. Сложен research project с различни аспекти
3. Document processing: extract → analyze → summarize

---

## Упражнение 11: Agent Reliability

Benchmark резултати на coding agent:

| Benchmark | Success Rate |
|-----------|--------------|
| Simple tasks | 85% |
| Medium tasks | 52% |
| Complex tasks | 23% |
| Multi-file refactoring | 8% |

**Въпроси:**
1. Защо има такъв спад при complexity?
2. Какво означава 8% за production use?
3. Как бихте подобрили reliability?

---

## Упражнение 12: Agent Cost Analysis

Task изисква средно:
- 5 LLM calls (reasoning + tool selection)
- 3 tool executions
- 2 retry attempts при грешки

**Costs:**
- GPT-4: $0.03/1K input + $0.06/1K output
- Среден input: 2K tokens, среден output: 500 tokens

**Задача:**
1. Каква е цената за един LLM call?
2. Каква е общата цена за task (5 calls)?
3. При 1000 tasks на ден, месечна цена?

---

# Решения

## Решение 1
1. Напиши имейл → A) Чист LLM
2. Времето в София → B) Agent (изисква real-time data)
3. Изчисли 847 × 293 → B) Agent (calculator tool за точност)
4. Обобщи документ → A) Чист LLM
5. Резервирай маса → B) Agent (изисква booking API)
6. Теория на относителността → A) Чист LLM

## Решение 2
Правилен ред: **E → C → D → A → B**
1. E) Получаване на user input
2. C) Reasoning (мислене)
3. D) Избор на tool/action
4. A) Изпълнение на action
5. B) Получаване на observation
(После цикълът се повтаря от C)

## Решение 3
1. Description е критичен защото LLM използва description за да реши кога да използва tool
2. Неясен description → модел избира грешен tool или не го използва когато трябва
3. Default стойност позволява simpler tool calls - моделът не трябва да решава за всеки параметър

## Решение 4
1. Multi-hop reasoning: директен query може да не даде резултат, защото search engine не прави inference
2. Thought стъпките правят reasoning explicit - помагат на модела да планира и да следи progress
3. Observation дава нова информация, която Thought интегрира за следващото действие

## Решение 5
1. "15% от 847" → calculator (или code_exec за по-сложно)
2. "Новини днес" → search
3. "Анализирай sales.csv" → file_read + code_exec (четене, после анализ)
4. "Функция за сортиране" → code_exec

## Решение 6
- Short-term → B) Текущият conversation history
- Long-term (episodic) → C) Запомня минали взаимодействия и техните резултати
- Long-term (semantic) → A) Запомня факти и знания между сесии

## Решение 7
Примерна декомпозиция:
1. Събиране на данни за climate change (search + data sources)
2. Анализ и избор на ключови статистики
3. Генериране на графики от данните (code_exec)
4. Структуриране в презентация

Директното изпълнение е проблематично защото:
- Твърде много стъпки наведнъж
- Няма checkpoint-и при грешки
- Трудно debug-ване

## Решение 8
1. 5 стъпки: $0.9^5 = 0.59$ (59% успех)
2. 10 стъпки: $0.9^{10} = 0.35$ (35% успех)
3. Фундаментален проблем: дори с висока per-step reliability, дългите планове имат ниска обща reliability. Това обяснява защо agents се справят зле с complex tasks.

## Решение 9
1. Reflection съхранява "lessons learned" - agent не повтаря същата грешка
2. В long-term memory (semantic или episodic)
3. Да - reflection може да направи грешни обобщения от единичен случай

## Решение 10
1. Code review → Debate (критика и подобрение чрез спор)
2. Research project → Hierarchical (manager разпределя research области)
3. Document processing → Pipeline (последователна обработка)

## Решение 11
1. Complexity води до повече стъпки → error accumulation; повече edge cases; по-трудно planning
2. 8% е неприемливо за production - 92% failure rate
3. Подобрения: по-добро planning, по-малки subtasks, human-in-the-loop за критични стъпки, по-добри tools

## Решение 12
1. Един LLM call: (2K × $0.03/1K) + (0.5K × $0.06/1K) = $0.06 + $0.03 = $0.09
2. 5 calls: 5 × $0.09 = $0.45 per task
3. 1000 tasks/day × 30 days × $0.45 = $13,500/месец

*Забележка: Agent-based systems могат да бъдат значително по-скъпи от single-call LLM applications!*
