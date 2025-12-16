# Математическа бележка: Линейна алгебра и вероятности

Тази бележка обобщава математическите основи, необходими за курса. Фокусът е върху интуиция и конкретни примери.

---

## 1. Вектори

### Какво е вектор?

Вектор е подредена последователност от числа.

$$\mathbf{v} = \begin{bmatrix} 3 \\ -1 \\ 4 \end{bmatrix} \in \mathbb{R}^3$$

Геометрично: стрелка от началото на координатната система до точка в пространството.

### Операции с вектори

**Събиране:** Елемент по елемент.

$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ -1 \end{bmatrix} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$$

Геометрично: слагаме втората стрелка в края на първата.

**Умножение със скалар:**

$$3 \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \begin{bmatrix} 6 \\ -3 \end{bmatrix}$$

Геометрично: разтягаме (или свиваме) стрелката.

### Скаларно произведение (dot product)

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

**Пример:**

$$\begin{bmatrix} 2 \\ 3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\ -1 \end{bmatrix} = 2 \cdot 4 + 3 \cdot (-1) = 8 - 3 = 5$$

**Геометрична интерпретация:**

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

където $\theta$ е ъгълът между векторите.

**Следствия:**
- Ако $\mathbf{a} \cdot \mathbf{b} > 0$: ъгълът е остър (векторите „сочат в сходна посока")
- Ако $\mathbf{a} \cdot \mathbf{b} = 0$: векторите са перпендикулярни
- Ако $\mathbf{a} \cdot \mathbf{b} < 0$: ъгълът е тъп

**Пример:** Проверка за перпендикулярност.

$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} -2 \\ 1 \end{bmatrix} = 1 \cdot (-2) + 2 \cdot 1 = 0$$

Да, перпендикулярни са.

### Норма (дължина)

$$\|\mathbf{v}\| = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

**Пример:**

$$\left\|\begin{bmatrix} 3 \\ 4 \end{bmatrix}\right\| = \sqrt{9 + 16} = 5$$

**Единичен вектор:** Вектор с норма 1. За да нормализираме вектор:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

### Косинусова прилика

За единични вектори скаларното произведение директно дава косинуса на ъгъла:

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

**Пример:**

$$\mathbf{a} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$\cos\theta = \frac{1 \cdot 1 + 0 \cdot 1}{\sqrt{1} \cdot \sqrt{2}} = \frac{1}{\sqrt{2}} \approx 0.707$$

Следователно $\theta = 45°$.

---

## 2. Матрици

### Какво е матрица?

Правоъгълна таблица от числа.

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

Тук $\mathbf{A}$ има 2 реда и 3 колони.

### Матрично умножение

За $\mathbf{A} \in \mathbb{R}^{m \times n}$ и $\mathbf{B} \in \mathbb{R}^{n \times p}$, произведението $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times p}$.

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

Всеки елемент е скаларно произведение на ред от $\mathbf{A}$ с колона от $\mathbf{B}$.

**Пример:**

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

**Важно:** $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ в общия случай (некомутативност).

### Матрица като линейна трансформация

Умножението $\mathbf{A}\mathbf{x}$ трансформира вектор $\mathbf{x}$ в нов вектор.

**Пример: Ротация на 90° обратно на часовниковата стрелка**

$$\mathbf{R} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

Прилагаме към $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$:

$$\mathbf{R}\mathbf{x} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Векторът се завърта от „надясно" към „нагоре".

**Пример: Мащабиране**

$$\mathbf{S} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$$

$$\mathbf{S}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

Разтяга по x-ос с фактор 2, по y-ос с фактор 3.

**Пример: Проекция върху права**

Проекция върху правата $y = x$:

$$\mathbf{P} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}$$

$$\mathbf{P}\begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

### Транспониране

Разменяме редове и колони.

$$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \implies \mathbf{A}^T = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix}$$

**Свойство:** $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$

### Обратна матрица

За квадратна матрица $\mathbf{A}$, обратната $\mathbf{A}^{-1}$ удовлетворява:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

където $\mathbf{I}$ е единичната матрица.

**Пример:** За $2 \times 2$ матрица:

$$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \implies \mathbf{A}^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

При условие че $\det(\mathbf{A}) = ad - bc \neq 0$.

**Пример:**

$$\mathbf{A} = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}, \quad \det(\mathbf{A}) = 24 - 14 = 10$$

$$\mathbf{A}^{-1} = \frac{1}{10}\begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}$$

**Геометрична интуиция:** Ако $\mathbf{A}$ ротира и мащабира, $\mathbf{A}^{-1}$ прави обратното.

### Собствени стойности и собствени вектори

Вектор $\mathbf{v} \neq \mathbf{0}$ е собствен вектор на $\mathbf{A}$ със собствена стойност $\lambda$, ако:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

Матрицата не променя посоката на $\mathbf{v}$, само го мащабира с $\lambda$.

**Пример:**

$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

Проверяваме $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$:

$$\mathbf{A}\mathbf{v} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix} = 3\begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

Да, $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ е собствен вектор с $\lambda = 3$.

Проверяваме $\mathbf{u} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$:

$$\mathbf{A}\mathbf{u} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \end{bmatrix} = 1 \cdot \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

Собствен вектор с $\lambda = 1$.

**Намиране на собствени стойности:** Решаваме $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$.

---

## 3. Вероятности

### Основни аксиоми

1. $P(A) \geq 0$ за всяко събитие $A$
2. $P(\Omega) = 1$ където $\Omega$ е цялото пространство
3. За несъвместими събития: $P(A \cup B) = P(A) + P(B)$

### Условна вероятност

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Пример:** В тесте карти. Каква е вероятността втората карта да е асо, ако първата е асо (без връщане)?

- $P(\text{първа е асо}) = \frac{4}{52}$
- $P(\text{втора е асо} | \text{първа е асо}) = \frac{3}{51}$

**Внимание:** $P(A|B) \neq P(B|A)$ в общия случай!

**Пример:**
- $P(\text{мокра трева} | \text{дъжд}) \approx 0.9$
- $P(\text{дъжд} | \text{мокра трева}) \approx 0.4$ (може и поливачката да е причина)

### Теорема на Байес

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Пример:** Медицински тест.
- Болест засяга 1% от популацията: $P(D) = 0.01$
- Тестът открива болестта в 99% от случаите: $P(+|D) = 0.99$
- Тестът дава фалшив положителен в 5% от здравите: $P(+|\neg D) = 0.05$

Ако тестът е положителен, каква е вероятността да сте болни?

$$P(D|+) = \frac{P(+|D) \cdot P(D)}{P(+)}$$

Първо намираме $P(+)$:

$$P(+) = P(+|D)P(D) + P(+|\neg D)P(\neg D) = 0.99 \cdot 0.01 + 0.05 \cdot 0.99 = 0.0594$$

$$P(D|+) = \frac{0.99 \cdot 0.01}{0.0594} \approx 0.167$$

Изненада: дори с положителен тест, вероятността да сте болни е само ~17%.

### Независимост

Събития $A$ и $B$ са независими, ако:

$$P(A \cap B) = P(A) \cdot P(B)$$

Еквивалентно: $P(A|B) = P(A)$.

**Пример:** Хвърляме две зарчета. Събитията „първото е 6" и „второто е четно" са независими.

$$P(\text{първо}=6 \cap \text{второ четно}) = \frac{1}{6} \cdot \frac{1}{2} = \frac{1}{12}$$

### Верижно правило за последователности (Chain Rule)

За последователност от събития $A_1, A_2, \ldots, A_n$:

$$P(A_1, A_2, \ldots, A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1, A_2) \cdots P(A_n|A_1, \ldots, A_{n-1})$$

**Приложение в езикови модели:** За последователност от думи $w_1, w_2, \ldots, w_n$:

$$P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, \ldots, w_{i-1})$$

**Markov предположение (N-gram):** Опростяваме, като гледаме само последните $k$ думи:

$$P(w_i | w_1, \ldots, w_{i-1}) \approx P(w_i | w_{i-k}, \ldots, w_{i-1})$$

**Пример (bigram, k=1):**

$$P(\text{the cat sat}) = P(\text{the}) \cdot P(\text{cat}|\text{the}) \cdot P(\text{sat}|\text{cat})$$

### Случайни величини

**Дискретна случайна величина:** Приема краен или изброим брой стойности.

**Пример:** Брой се успехи при 10 хвърляния на монета.

**Очаквана стойност (математическо очакване):**

$$E[X] = \sum_x x \cdot P(X = x)$$

**Пример:** Печалба от хазартна игра.

| Изход | Вероятност | Печалба |
|-------|------------|---------|
| Победа | 0.1 | 90 лв |
| Загуба | 0.9 | -10 лв |

$$E[\text{печалба}] = 0.1 \cdot 90 + 0.9 \cdot (-10) = 9 - 9 = 0$$

Играта е „честна" (в дългосрочен план нито печелиш, нито губиш).

**Дисперсия:** Мярка за разпръснатост.

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Стандартно отклонение:** $\sigma = \sqrt{\text{Var}(X)}$

### Често срещани разпределения

**Бернули:** Един опит с два изхода (успех/неуспех).

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$
$$E[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Биномно:** Брой успехи в $n$ независими Бернули опита.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Пример:** Вероятност за точно 3 ези от 5 хвърляния на честна монета:

$$P(X = 3) = \binom{5}{3} \cdot 0.5^3 \cdot 0.5^2 = 10 \cdot 0.03125 = 0.3125$$

**Нормално (Гаусово):** Непрекъснато разпределение с плътност:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Свойства:
- Симетрично около средната стойност $\mu$
- ~68% от данните са в интервала $[\mu - \sigma, \mu + \sigma]$
- ~95% са в $[\mu - 2\sigma, \mu + 2\sigma]$

---

## 4. Производни и градиенти

### Производна

Производната на $f(x)$ показва скоростта на изменение:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Основни правила:**

| $f(x)$ | $f'(x)$ |
|--------|---------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sin x$ | $\cos x$ |

**Верижно правило:** За композиция $f(g(x))$:

$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

**Пример:**

$$\frac{d}{dx} e^{x^2} = e^{x^2} \cdot 2x$$

### Частни производни

За функция на няколко променливи, частната производна по $x_i$ третира другите като константи.

**Пример:** $f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy$$

### Градиент

Градиентът е вектор от всички частни производни:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Ключово свойство:** Градиентът сочи в посоката на най-бърз растеж на функцията.

**Пример:** $f(x, y) = x^2 + y^2$

$$\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}$$

В точка $(1, 2)$: $\nabla f = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$

Функцията расте най-бързо в посока $(2, 4)$ (или нормализирано: $(\frac{1}{\sqrt{5}}, \frac{2}{\sqrt{5}})$).

### Критични точки

В минимум или максимум: $\nabla f = \mathbf{0}$.

**Пример:** Намерете минимума на $f(x, y) = (x-1)^2 + (y+2)^2$.

$$\nabla f = \begin{bmatrix} 2(x-1) \\ 2(y+2) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Решение: $x = 1$, $y = -2$. Минимумът е в точка $(1, -2)$.

---

## 5. Логаритми и експоненти

### Основни свойства

$$e^{a+b} = e^a \cdot e^b$$
$$\ln(ab) = \ln a + \ln b$$
$$\ln(a^b) = b \ln a$$
$$e^{\ln x} = x$$
$$\ln(e^x) = x$$

### Защо логаритми?

**Проблем:** Произведение на много малки числа бързо клони към 0.

$$0.01 \times 0.02 \times 0.03 \times \cdots \approx 0$$

Компютрите не могат да представят толкова малки числа (underflow).

**Решение:** Работим с логаритми.

$$\ln(0.01 \times 0.02 \times 0.03) = \ln(0.01) + \ln(0.02) + \ln(0.03)$$
$$\approx -4.6 + (-3.9) + (-3.5) = -12$$

Сумата е числено стабилна.

### Cross-Entropy и Perplexity

**Cross-entropy** измерва колко добре вероятностно разпределение $Q$ апроксимира истинското $P$:

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

**За езикови модели:** Ако $P$ е емпиричното разпределение (1 за истинската следваща дума, 0 за останалите):

$$H = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1})$$

**Perplexity** е експонента на cross-entropy:

$$\text{Perplexity} = 2^{H} = \left( \prod_{i=1}^{N} P(w_i | \text{context}) \right)^{-1/N}$$

**Интуиция:** Ако perplexity = 100, моделът е „изненадан" колкото при избор от 100 равновероятни думи.

**По-ниска perplexity = по-добър модел.**

### Log-sum-exp трик

Искаме да изчислим $\ln(e^{a} + e^{b})$ когато $a$ и $b$ са големи (или много отрицателни).

**Проблем:** $e^{1000}$ предизвиква overflow.

**Решение:** Нека $m = \max(a, b)$.

$$\ln(e^a + e^b) = m + \ln(e^{a-m} + e^{b-m})$$

Сега $a - m \leq 0$ и $b - m \leq 0$, така че експонентите не експлодират.

**Пример:** $a = 1000$, $b = 1001$.

$$\ln(e^{1000} + e^{1001}) = 1001 + \ln(e^{-1} + 1) = 1001 + \ln(1.368) \approx 1001.31$$

---

## 6. Препоръчителни ресурси

**Линейна алгебра:**
- 3Blue1Brown: „Essence of Linear Algebra" (YouTube)
- Gilbert Strang: Linear Algebra (MIT OCW)
- „Linear Algebra Done Right" — Sheldon Axler

**Вероятности:**
- „Introduction to Probability" — Blitzstein & Hwang (безплатна онлайн)
- Khan Academy: Probability and Statistics

**Калкулус:**
- 3Blue1Brown: „Essence of Calculus" (YouTube)
- „Calculus Made Easy" — Silvanus Thompson (класика, безплатна)

**Обща математика за ML:**
- „Mathematics for Machine Learning" — Deisenroth, Faisal, Ong (безплатна PDF)
