# Эксперименты и результаты

Экспериментальная часть проверяет, можно ли корректно сравнивать классификационные, регрессионные и survival-модели после приведения их откликов к функции выживания.

## Датасеты

Использовались семь открытых медицинских наборов данных из `survivors.datasets`.

| Датасет | N наблюдений | N признаков | Целевые переменные | Цензурирование | Событие |
| --- | ---: | ---: | --- | ---: | --- |
| ACTG | 1151 | 11 | `time`, `event` | 91.66% | СПИД |
| GBSG | 686 | 8 | `rfst`, `cens` | 56.41% | рецидив рака молочной железы |
| PBC | 418 | 17 | `time`, `status` | 61.48% | смерть |
| Rott2 | 2982 | 11 | `time`, `event` | 57.34% | смерть |
| Smarto | 3873 | 26 | `TEVENT`, `EVENT` | 88.12% | сердечно-сосудистое заболевание |
| Framingham | 4699 | 7 | `followup`, `chdfate` | 68.35% | ишемическая болезнь сердца |
| Support2 | 9105 | 35 | `d.time`, `death` | 31.89% | смерть |

ACTG и Smarto особенно важны для проверки survival-подходов, потому что имеют очень высокую долю цензурирования.

## Использованные модели

В эксперименте сравнивались 19 методов.

Классификация:

- LogisticRegression;
- SVC;
- KNeighborsClassifier;
- DecisionTreeClassifier;
- RandomForestClassifier;
- GradientBoostingClassifier.

Регрессия:

- ElasticNet;
- SVR;
- KNeighborsRegressor;
- DecisionTreeRegressor;
- RandomForestRegressor;
- GradientBoostingRegressor.

Анализ выживаемости:

- KaplanMeierFitter;
- CoxPHSurvivalAnalysis;
- SurvivalTree;
- RandomSurvivalForest;
- GradientBoostingSurvivalAnalysis;
- CRAID;
- ParallelBootstrapCRAID.

Таблица "Использованные модели" группирует методы по постановке и классу:

| Класс моделей | Классификация | Регрессия | Принцип работы |
| --- | --- | --- | --- |
| Линейные | Logistic Regression | Linear Regression, Elastic Net | используют линейную зависимость прогноза от признаков объекта |
| Метрические | k-Nearest Neighbors, Support Vector Classifier | k-Nearest Neighbors Regressor, Support Vector Regression | основаны на мере близости объектов в пространстве признаков |
| Древовидные | Decision Tree, Random Forest, Gradient Boosting | Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor | последовательно разбивают пространство признаков и формируют прогноз по листьям |

| Класс survival-моделей | Модели | Принцип работы |
| --- | --- | --- |
| Непараметрические | Kaplan-Meier | оценивают функцию выживаемости по наблюдаемым временам событий без признаков |
| Полупараметрические | Cox Proportional Hazards | оценивают влияние признаков на риск при предположении постоянного отношения рисков |
| Древовидные | Survival Tree, CRAID | разбивают пространство признаков на группы с различной структурой выживаемости |
| Ансамблевые | Random Survival Forest, Gradient Boosting Survival Analysis, Parallel Bootstrap CRAID | объединяют несколько survival-моделей для устойчивой оценки риска и функции выживаемости |

## Piecewise-times модели

Дополнительно исследовались piecewise-адаптеры для классификации:

- `PiecewiseClassifWrapSA`;
- `PiecewiseCensorAwareClassifWrapSA`.

В текущей версии интерфейса и итогового leaderboard используется единая временная сетка `times=16`.

Piecewise-times строит интервальные классификаторы и преобразует их в survival-кривую:

```text
S_hat(t_k | X) = prod_{j=1}^{k} (1 - p_j(X))
```

По classification score, где `AUC_EVENT` имеет вес 45%, `LOGLOSS_EVENT` - 35%, `RMSE_EVENT` - 20%, самый сильный эффект при `times=16` получен для `DecisionTreeClassifier`: средний прирост +26.8 пункта по 7 датасетам. Улучшение для `DecisionTreeClassifier` наблюдалось на всех 7 датасетах; лучший кейс - GBSG, +53.7 пункта при `times=16`.

## Метрики в исследовании

Классификационный блок:

- `AUC_EVENT`: выше лучше;
- `LOGLOSS_EVENT`: ниже лучше;
- `RMSE_EVENT`: ниже лучше.

Регрессионный блок:

- `RMSE_TIME`: ниже лучше;
- `R2_TIME`: выше лучше;
- `MAPE_TIME`: ниже лучше;
- `MEDAPE_TIME`: ниже лучше;
- `SPEARMAN_TIME`: выше лучше;
- `RMSLE_TIME`: ниже лучше.

Survival-блок:

- `CI`: выше лучше;
- `IBS`: ниже лучше;
- `AUPRC`: выше лучше.

Таблица "Метрики в исследовании" включает все 12 метрик из диплома:

| Постановка | Метрика | Формула | Краткое описание |
| --- | --- | --- | --- |
| Событие | `AUC_EVENT` | `AUC = 1/(n1*n0) * sum_{i:Y_i=1} sum_{j:Y_j=0} I(P_hat_i > P_hat_j)` | площадь под ROC-кривой |
| Событие | `LOGLOSS_EVENT` | `LogLoss = -1/n * sum_i [Y_i ln(P_hat_i) + (1-Y_i) ln(1-P_hat_i)]` | логарифмическая функция потерь |
| Событие | `RMSE_EVENT` | `RMSE = sqrt(1/n * sum_i (Y_i - P_hat_i)^2)` | среднеквадратичная ошибка вероятности события |
| Время | `RMSE_TIME` | `RMSE = sqrt(1/n * sum_i (T_i - T_hat_i)^2)` | корень из среднеквадратичной ошибки времени |
| Время | `R2_TIME` | `R2 = 1 - sum_i(T_i - T_hat_i)^2 / sum_i(T_i - T_mean)^2` | коэффициент детерминации |
| Время | `MAPE_TIME` | `MAPE = 100/n * sum_i abs((T_i - T_hat_i) / max(T_i, 1))` | средняя абсолютная процентная ошибка |
| Время | `MEDAPE_TIME` | `MEDAPE = 100 * median abs((T_i - T_hat_i) / max(T_i, 1))` | медианная абсолютная процентная ошибка |
| Время | `SPEARMAN_TIME` | `rho_s = 1 - 6 * sum_i d_i^2 / (n(n^2 - 1))` | коэффициент ранговой корреляции Спирмена |
| Время | `RMSLE_TIME` | `RMSLE = sqrt(1/n * sum_i (ln(1+T_i) - ln(1+T_hat_i))^2)` | логарифмическая среднеквадратичная ошибка |
| Выживаемость | `CI` | `CI = число согласованных пар / число сравнимых пар` | индекс согласованности |
| Выживаемость | `IBS` | `IBS = 1/t_max * int_0^{t_max} BS(t) dt` | интегрированная оценка Брайера |
| Выживаемость | `AUPRC` | `AUPRC = int_0^1 P(T_i/phi > T > T_i phi) dphi` | площадь под survival precision-recall curve |

## Формулы ключевых метрик

В дипломе формулы метрик приведены в разделе с методами оценки качества. Основные определения:

- `AUC_EVENT`: \(\mathrm{AUC}=\frac{1}{n_1n_0}\sum_{i:Y_{\tau,i}=1}\sum_{j:Y_{\tau,j}=0}\left[\mathbb{I}(\widehat P_i>\widehat P_j)+\frac{1}{2}\mathbb{I}(\widehat P_i=\widehat P_j)\right]\).
- `LOGLOSS_EVENT`: \(\mathrm{LogLoss}=-\frac{1}{n}\sum_{i=1}^{n}\left[Y_{\tau,i}\ln \widehat P_i+(1-Y_{\tau,i})\ln(1-\widehat P_i)\right]\).
- `RMSE_EVENT`: \(\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(Y_{\tau,i}-\widehat P_i)^2}\).
- `RMSE_TIME`: \(\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\widetilde T_i-\widehat T(X_i))^2}\).
- `R2_TIME`: \(R^2=1-\frac{\sum_{i=1}^{n}(\widetilde T_i-\widehat T(X_i))^2}{\sum_{i=1}^{n}(\widetilde T_i-\overline T)^2}\).
- `MAPE_TIME`: \(\mathrm{MAPE}=\frac{100}{n}\sum_{i=1}^{n}\left|\frac{\widetilde T_i-\widehat T(X_i)}{\max(\widetilde T_i,1)}\right|\).
- `MEDAPE_TIME`: \(\mathrm{MEDAPE}=100\cdot\mathrm{median}\left|\frac{\widetilde T_i-\widehat T(X_i)}{\max(\widetilde T_i,1)}\right|\).
- `SPEARMAN_TIME`: \(\rho_s=1-\frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)}\), если нет совпадающих рангов.
- `RMSLE_TIME`: \(\mathrm{RMSLE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\ln(1+\widetilde T_i)-\ln(1+\widehat T(X_i))\right)^2}\).
- `CI`: \(\mathrm{CI}=\frac{\sum_{i,j}\mathbb{I}(\widetilde T_j<\widetilde T_i)\mathbb{I}(\widehat T(X_j)<\widehat T(X_i))}{\sum_{i,j}\mathbb{I}(\widetilde T_j<\widetilde T_i)}\).
- `IBS`: \(\mathrm{IBS}=\frac{1}{t_{\max}}\int_{0}^{t_{\max}}BS(t)\,dt\).
- `AUPRC`: \(AUPRC_{\delta=1}(\widehat S,T_i)=\int_{0}^{1}P(T_i/\phi>T>T_i\phi)\,d\phi\).

## Протокол эксперимента

- Исходный набор разбивался на обучающую и тестовую выборки.
- Прогнозы моделей адаптировались к единой постановке.
- Лучшие гиперпараметры выбирались по сетке на 5-кратной кросс-валидации на обучающей выборке для всех рассматриваемых моделей.
- Лучшие модели обучались и валидировались в едином экспериментальном сценарии на 20 различных разбиениях данных.
- Модели сравнивались по метрикам трех типов: событие, время события и функция выживания.
- Итоговая таблица лидеров строилась через ранжирование моделей по значениям метрик и агрегирование рангов по наборам данных и постановкам по медианному рангу.
- Общий leaderboard по всем постановкам формировался по среднему рангу.

## Итоговое ранжирование

Лучшие модели по итоговому агрегированному сравнению:

| Место | Модель | Событие | Время | Выживаемость | Средняя позиция |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | ParallelBootstrapCRAID | 7 | 4 | 1 | 4.00 |
| 2 | CRAID | 9 | 6 | 2 | 5.67 |
| 3 | RandomForestRegressor | 11 | 1 | 7 | 6.33 |
| 4 | GradientBoostingSurvivalAnalysis | 4 | 13 | 3 | 6.67 |
| 5 | GradientBoostingRegressor | 14 | 2 | 8 | 8.00 |
| 6 | ElasticNet | 12 | 4 | 9 | 8.33 |
| 7 | RandomForestClassifier | 2 | 10 | 13 | 8.33 |
| 8 | CoxPHSurvivalAnalysis | 8 | 14 | 4 | 8.67 |
| 9 | LogisticRegression | 4 | 10 | 13 | 9.00 |
| 10 | GradientBoostingClassifier | 2 | 11 | 15 | 9.33 |

## Интерпретация

Главный вывод: лучшие позиции заняли преимущественно модели анализа выживаемости и ансамблевые методы.

`ParallelBootstrapCRAID` оказался лучшим по общей устойчивости, потому что занял первое место в survival-блоке и сохранил сильные позиции в прогнозе времени.

`CRAID` также показал высокое качество в survival-задаче и устойчивое поведение в разных постановках.

`RandomForestRegressor` особенно силен в прогнозировании времени события, что показывает, что точечный прогноз времени может сохранять полезную информацию после преобразования к survival-представлению.

Классификационные модели полезны для прогноза факта события, но хуже отражают временную структуру и цензурирование.
