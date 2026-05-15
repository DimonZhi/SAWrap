# Data Science

Раздел описывает Data Science-часть проекта: EDA, предобработку, выбор моделей, метрики и валидацию.

## EDA и данные

В проекте используются семь открытых медицинских survival-датасетов: ACTG, GBSG, PBC, Rott2, Smarto, Framingham и Support2. Для них учитываются размер выборки, число признаков, объект прогноза и доля цензурирования.

Доля цензурирования важна, потому что обычная регрессия и классификация теряют часть информации, когда событие не наблюдалось до конца периода наблюдения.

## Предобработка

Данные приводятся к time-to-event представлению: признаки объекта, наблюдаемое время и индикатор события. После этого классификационные, регрессионные и survival-модели можно привести к общей функции выживания.

## Выбор моделей

Сравниваются три семейства:

- классификация: LogisticRegression, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier;
- регрессия: ElasticNet, SVR, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor;
- survival analysis: KaplanMeierFitter, CoxPHSurvivalAnalysis, SurvivalTree, RandomSurvivalForest, GradientBoostingSurvivalAnalysis, CRAID, ParallelBootstrapCRAID.

Piecewise-модели планируется добавить позже как отдельное расширение survival-семейства.

## Метрики

Используются 12 метрик:

- событие: AUC_EVENT, LOGLOSS_EVENT, RMSE_EVENT;
- время: RMSE_TIME, R2_TIME, MAPE_TIME, MEDAPE_TIME, SPEARMAN_TIME, RMSLE_TIME;
- выживаемость: CI, IBS, AUPRC.

Метрики имеют разные направления: для AUC, R2, Spearman, CI и AUPRC больше лучше; для LogLoss, RMSE, MAPE, MedAPE, RMSLE и IBS меньше лучше.

## Валидация

Используется 5-кратная кросс-валидация, подбор гиперпараметров и серия из 20 разбиений. Затем модели ранжируются внутри датасетов и блоков метрик, после чего строится итоговый лидерборд.
