# Эксперименты и результаты

Экспериментальная часть проверяет, можно ли корректно сравнивать классификационные, регрессионные и survival-модели после приведения их откликов к функции выживания.

## Датасеты

Использовались семь открытых медицинских наборов данных из `survivors.datasets`.

| Датасет | Объект прогноза | Размер | Признаки | Цензурирование |
| --- | --- | ---: | ---: | ---: |
| ACTG | развитие СПИДа | 1151 | 11 | 91.66% |
| GBSG | рецидив рака молочной железы | 686 | 8 | 56.41% |
| PBC | летальный исход | 418 | 17 | 61.48% |
| Rott2 | летальный исход | 2982 | 11 | 57.34% |
| Smarto | сердечно-сосудистое заболевание | 3873 | 26 | 88.12% |
| Framingham | ишемическая болезнь сердца | 4699 | 7 | 68.35% |
| Support2 | летальный исход | 9105 | 35 | 31.89% |

ACTG и Smarto особенно важны для проверки survival-подходов, потому что имеют очень высокую долю цензурирования.

## Модели

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

## Метрики

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

## Формулы ключевых метрик

В дипломе формулы метрик приведены в разделе с методами оценки качества. Основные определения:

- `AUC_EVENT`: \(\mathrm{AUC}=\frac{1}{n_1n_0}\sum_{i:Y_{\tau,i}=1}\sum_{j:Y_{\tau,j}=0}\left[\mathbb{I}(\widehat P_i>\widehat P_j)+\frac{1}{2}\mathbb{I}(\widehat P_i=\widehat P_j)\right]\).
- `LOGLOSS_EVENT`: \(\mathrm{LogLoss}=-\frac{1}{n}\sum_{i=1}^{n}\left[Y_{\tau,i}\ln \widehat P_i+(1-Y_{\tau,i})\ln(1-\widehat P_i)\right]\).
- `RMSE_EVENT`: \(\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(Y_{\tau,i}-\widehat P_i)^2}\).
- `RMSE_TIME`: \(\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\widetilde T_i-\widehat T(X_i))^2}\).
- `R2_TIME`: \(R^2=1-\frac{\sum_{i=1}^{n}(\widetilde T_i-\widehat T(X_i))^2}{\sum_{i=1}^{n}(\widetilde T_i-\overline T)^2}\).
- `MAPE_TIME`: \(\mathrm{MAPE}=\frac{100}{n}\sum_{i=1}^{n}\left|\frac{\widetilde T_i-\widehat T(X_i)}{\max(\widetilde T_i,1)}\right|\).
- `SPEARMAN_TIME`: \(\rho_s=1-\frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)}\), если нет совпадающих рангов.
- `CI`: \(\mathrm{CI}=\frac{\sum_{i,j}\mathbb{I}(\widetilde T_j<\widetilde T_i)\mathbb{I}(\widehat T(X_j)<\widehat T(X_i))}{\sum_{i,j}\mathbb{I}(\widetilde T_j<\widetilde T_i)}\).
- `IBS`: \(\mathrm{IBS}=\frac{1}{t_{\max}}\int_{0}^{t_{\max}}BS(t)\,dt\).
- `AUPRC`: \(AUPRC_{\delta=1}(\widehat S,T_i)=\int_{0}^{1}P(T_i/\phi>T>T_i\phi)\,d\phi\).

## Протокол эксперимента

- Данные разбивались на обучающую и тестовую выборки.
- Для моделей выполнялся подбор гиперпараметров по сетке.
- Использовалась 5-кратная кросс-валидация.
- Лучшие конфигурации проверялись на серии из 20 разбиений.
- Для каждой модели вычислялись 12 метрик.
- Модели ранжировались внутри каждого датасета и блока метрик.
- Затем строилось итоговое ранжирование по всем датасетам и трем постановкам.

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
