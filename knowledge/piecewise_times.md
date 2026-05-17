# Piecewise-times расширение

Piecewise-times - это инженерное расширение классификационного адаптера SAWrap. Обычный `ClassifWrapSA` строит постоянную по времени функцию выживания:

```text
S_hat(t | X) = 1 - P_hat(X)
```

Такая схема дает единый survival-интерфейс, но классификатор все еще не различает ранние и поздние события.

Piecewise-подход делит временной горизонт на интервалы и обучает отдельный классификационный риск для каждого интервала. В текущей версии сайта и leaderboard система сначала выбирает один лучший `times` для каждой пары "Piecewise-обертка + базовый классификатор" по всем датасетам, а затем использует только эту вариацию в результатах и графиках.

Если `p_j(X)` - вероятность события в j-м интервале, то survival-прогноз строится как произведение вероятностей пережить интервалы:

```text
S_hat(t_k | X) = prod_{j=1}^{k} (1 - p_j(X))
```

## Реализованные варианты

- `PiecewiseClassifWrapSA` - базовый piecewise-вариант для классификаторов.
- `PiecewiseCensorAwareClassifWrapSA` - вариант, учитывающий цензурирование при построении интервальных задач.

Оба варианта сохраняют идею проекта: базовая модель остается классификатором, но ее прогноз переводится в более информативную survival-кривую.

## Почему это сильная разработка

Piecewise-times закрывает слабое место обычной классификации. Вместо одного прогноза события на весь горизонт модель получает временную структуру и может различать интервалы риска. Это особенно полезно для простых классификаторов, которые без piecewise-схемы теряют информацию о времени.

## Результаты по classification score

Classification score считается по тем же весам, что AI-интерпретатор:

- `AUC_EVENT`: 45%;
- `LOGLOSS_EVENT`: 35%;
- `RMSE_EVENT`: 20%.

На таблицах `UI/tables/Piecewise_*.xlsx` и `UI/tables/Piesewise_pbc.xlsx` Piecewise особенно сильно улучшил `DecisionTreeClassifier` при глобальном выборе одного `times` на Piecewise-модель:

| Показатель | Значение |
| --- | ---: |
| Средний прирост classification score для двух Piecewise-вариантов `DecisionTreeClassifier` | +24.1 пункта |
| Пар датасет/вариант с улучшением `DecisionTreeClassifier` | 13 из 14 |
| Лучший кейс | GBSG, +53.7 пункта |
| Выбор вариации | один глобальный `times` для каждой пары обертка + базовый классификатор |

Глобально выбранные варианты для `DecisionTreeClassifier`:

| Piecewise-модель | Выбранный times | Датасетов | Средний прирост | Win-rate |
| --- | ---: | ---: | ---: | ---: |
| `PiecewiseClassifWrapSA(DecisionTreeClassifier)` | 8 | 7 | +24.8 | 100% |
| `PiecewiseCensorAwareClassifWrapSA(DecisionTreeClassifier)` | 16 | 7 | +23.4 | 86% |

## Новые агрегированные Piecewise-результаты

После перехода на глобальный выбор `times` каждая Piecewise-модель оценивается одной и той же временной вариацией на всех 7 датасетах. Это убирает ситуацию, когда на разных датасетах одна и та же модель появлялась в leaderboard с разными `times`.

| Piecewise-модель | Выбранный times | Датасетов | Base score | Piecewise score | Прирост | Win-rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PiecewiseClassifWrapSA(DecisionTreeClassifier)` | 8 | 7 | 41.5 | 66.3 | +24.8 | 100% |
| `PiecewiseCensorAwareClassifWrapSA(DecisionTreeClassifier)` | 16 | 7 | 41.5 | 64.9 | +23.4 | 86% |
| `PiecewiseClassifWrapSA(KNeighborsClassifier)` | 8 | 7 | 40.4 | 45.0 | +4.6 | 71% |
| `PiecewiseCensorAwareClassifWrapSA(KNeighborsClassifier)` | 8 | 7 | 40.4 | 43.3 | +2.9 | 57% |
| `PiecewiseClassifWrapSA(LogisticRegression)` | 8 | 7 | 92.0 | 86.6 | -5.4 | 14% |
| `PiecewiseCensorAwareClassifWrapSA(LogisticRegression)` | 16 | 7 | 92.0 | 83.5 | -8.5 | 14% |
| `PiecewiseClassifWrapSA(RandomForestClassifier)` | 8 | 7 | 96.4 | 87.2 | -9.2 | 0% |
| `PiecewiseClassifWrapSA(GradientBoostingClassifier)` | 4 | 7 | 97.0 | 86.0 | -11.0 | 14% |
| `PiecewiseCensorAwareClassifWrapSA(RandomForestClassifier)` | 16 | 7 | 96.4 | 80.4 | -16.0 | 0% |
| `PiecewiseCensorAwareClassifWrapSA(GradientBoostingClassifier)` | 16 | 7 | 97.0 | 77.1 | -20.0 | 0% |

В общем leaderboard по всем постановкам задачи Piecewise-варианты сохраняют покрытие всех 7 датасетов. Лучший Piecewise-результат в `OVERALL_ALL`:

| Место | Метод | Датасетов | Средний агрегированный ранг |
| ---: | --- | ---: | ---: |
| 3 | `PiecewiseCensorAwareClassifWrapSA(LogisticRegression, times=16)` | 7 | 9.0 |
| 5 | `PiecewiseCensorAwareClassifWrapSA(RandomForestClassifier, times=16)` | 7 | 9.7 |
| 6 | `PiecewiseClassifWrapSA(RandomForestClassifier, times=8)` | 7 | 10.0 |
| 7 | `PiecewiseCensorAwareClassifWrapSA(GradientBoostingClassifier, times=16)` | 7 | 10.7 |
| 8 | `PiecewiseClassifWrapSA(LogisticRegression, times=8)` | 7 | 11.7 |

Общий вывод: Piecewise-times не обязан улучшать каждый классификатор, но для `DecisionTreeClassifier` эффект устойчивый и крупный. Это можно использовать как сильный инженерный вклад проекта: была найдена и реализована схема, которая превращает простой классификационный подход в более временно-информативный survival-прогноз.
