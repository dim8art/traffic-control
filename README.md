# Traffic Control

[![CI](https://github.com/dim8art/traffic-control/actions/workflows/ci.yml/badge.svg)](https://github.com/dim8art/traffic-control/actions/workflows/ci.yml)

Мультиагентное управление светофорами в [SUMO](https://eclipse.dev/sumo/): подготовка сети из OSM, обучение PPO (Ray RLlib), демо, сравнение с классическими режимами TLS и интерактивное меню.

## Возможности

- **Подготовка карты** — локальный OSM/PBF/GraphML → обрезка (bbox, полигон, пресеты районов СПб) → `.net.xml` для SUMO
- **Обучение** — PPO с несколькими агентами (по одному на TLS), Ray Tune, возобновление с чекпоинта
- **Симуляция** — прогон обученной политики или SUMO без RL (программа из сети)
- **Benchmark** — PPO vs static / actuated / delay-based / MaxPressure / Greedy
- **TUI** — пошаговое меню, если запустить `main.py` без аргументов
- Опционально: пешеходы, автобусы, несколько вариантов целевой функции вознаграждения, saliency наблюдений

## Требования

- **SUMO ≥ 1.18** (рекомендуется PPA `sumo/stable` на Ubuntu или `brew install sumo` на macOS)
- **Python 3.11+** (в CI — 3.12)
- Для `.pbf`: утилита [osmium-tool](https://osmcode.org/osmium-tool/)
- Для подготовки карт: GDAL/GEOS (ставятся скриптом `build.sh` или Docker-образом)

Версии пакетов **`traci` / `sumolib` на PyPI должны совпадать** с версией бинарника `sumo` — иначе возможны ошибки TraCI. Скрипт `build.sh` и Docker-образ подбирают их автоматически.

## Быстрый старт

### Локально (Linux / macOS)

```bash
git clone https://github.com/dim8art/traffic-control.git
cd traffic-control
./build.sh          # SUMO, venv, traci/sumolib, requirements.txt
source venv/bin/activate
export SUMO_HOME=/usr/share/sumo   # если ещё не в ~/.bashrc
```

### Docker

```bash
docker compose build
docker compose run --rm traffic-control --help
```

Каталоги `data/`, `ray_results/`, `checkpoints/` монтируются с хоста (см. `docker-compose.yml`).

## CLI

Единая точка входа — `main.py` (или `python main.py` в venv / контейнере):

| Команда | Назначение |
|---------|------------|
| `prepare` | Локальная карта → SUMO `.net.xml` |
| `train` | Обучение PPO |
| `demo` | Запуск обученной политики в SUMO |
| `sumo` | Симуляция без RL |
| `benchmark` | Сравнение режимов TLS |
| `tui` | Интерактивное меню |

Без аргументов открывается TUI:

```bash
python main.py
```

### Примеры

**Подготовка сети** (пресет района + быстрый режим netconvert):

```bash
python main.py prepare \
  --map /path/to/region.osm.pbf \
  --name kolomna \
  --area kolomna \
  --fast
```

Пресеты `--area`: `kolomna`, `vasilevsky`, `petrogradsky`, `center_admiralty`, `moskovsky`, `primorsky`, `frunzensky`, `vyborgsky`, `nevsky_district`, `kalininsky`, `krestovsky`, `kanonersky` и др. (см. `prepare_map.py`).

**Обучение:**

```bash
python main.py train \
  --map data/sumo/kolomna.net.xml \
  --period 0.5 \
  --duration 3600 \
  --training-iterations 50 \
  --with-pedestrians \
  --reward-mode pressure
```

**Демо с GUI:**

```bash
python main.py demo \
  --checkpoint ray_results/.../checkpoint_000001 \
  --map data/sumo/kolomna.net.xml
```

**Benchmark:**

```bash
python main.py benchmark \
  --checkpoint ray_results/.../checkpoint_000001 \
  --map data/sumo/kolomna.net.xml \
  --output results.json

# Несколько прогонов (случайный трафик): среднее, СКО и 95% ДИ в логе и JSON
python main.py benchmark \
  --checkpoint ray_results/.../checkpoint_000001 \
  --map data/sumo/kolomna.net.xml \
  --runs 5 \
  --confidence-level 0.95 \
  --output results_multi.json
```

Справка по подкоманде: `python main.py train --help`.

### Режимы награды (`--reward-mode`)

`pressure`, `inbound_queue`, `lane_delay`, а также варианты с пешеходами: `*_sidewalk` (см. `REWARD_MODES` в `src/simulation/env.py`).

Значение **`all`** (в TUI — «Все режимы подряд») запускает **6 отдельных** обучений подряд — для каждого `reward_mode` свой каталог в `ray_results/`. Продолжение с `--checkpoint` в этом режиме не используется.

## Структура проекта

```
traffic-control/
├── main.py              # Единый CLI
├── prepare_map.py       # Подготовка OSM → SUMO
├── train.py             # Ray Tune + PPO
├── demo.py              # Inference / SUMO без RL
├── benchmark.py         # Сравнение алгоритмов TLS
├── tui.py               # Интерактивное меню
├── build.sh             # Установка окружения
├── src/
│   ├── map_engine/      # OSMnx, netconvert
│   ├── simulation/      # SUMO runner, multi-agent env, MaxPressure/Greedy
│   └── rl/              # Saliency наблюдений
├── tests/               # pytest
├── data/                # network/, sumo/ (не в git)
└── .github/workflows/   # CI
```

## Результаты обучения

Каталог по умолчанию: `ray_results/`. Имя эксперимента формируется автоматически, например:

`20260522_153045_vasilevsky_p0p5_d3600_rmpressure_iter50_env6_rf50_pedauto`

Структура: `ray_results/<имя>/train/checkpoint_000…` (дата-время, карта, период, длительность, `reward_mode`, число итераций, воркеры, пешеходы/ОТ при включении). При продолжении с `--checkpoint` используется существующий каталог trial.

## Тесты и CI

```bash
python -m pytest tests/ -q
```

На GitHub Actions (ветка `main`): установка SUMO, зависимостей, `pytest` и сборка Docker-образа — см. [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `SUMO_HOME` | Каталог установки SUMO (например `/usr/share/sumo`) |
| `TRAFFIC_SUMO_PREFLIGHT` | `0` — не запускать headless-проверку `sim.sumocfg` перед каждым reset (быстрее при обучении) |

## Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
