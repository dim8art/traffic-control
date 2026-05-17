#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 1. Системные зависимости (SUMO, venv, GDAL для geopandas)
if [[ "${OSTYPE:-}" == linux-gnu* ]]; then
    echo "Установка системных зависимостей (Linux)..."
    sudo apt-get update
    sudo apt-get install -y \
        sumo sumo-tools sumo-doc \
        python3 python3-pip python3-venv \
        gdal-bin libgdal-dev libgeos-dev libproj-dev
elif [[ "${OSTYPE:-}" == darwin* ]]; then
    echo "Установка системных зависимостей (macOS)..."
    brew install sumo
else
    echo "Предупреждение: неизвестная ОС (${OSTYPE:-}), пропуск apt/brew." >&2
fi

# 2. SUMO_HOME
if [[ -z "${SUMO_HOME:-}" ]]; then
    echo "Настройка SUMO_HOME..."
    if [[ "${OSTYPE:-}" == linux-gnu* ]]; then
        export SUMO_HOME="/usr/share/sumo"
    elif [[ "${OSTYPE:-}" == darwin* ]]; then
        export SUMO_HOME="/usr/local/opt/sumo/share/sumo"
    fi
    if [[ -n "${SUMO_HOME:-}" ]]; then
        for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
            if [[ -f "$rc" ]] && ! grep -qF "SUMO_HOME=$SUMO_HOME" "$rc" 2>/dev/null; then
                echo "export SUMO_HOME=$SUMO_HOME" >>"$rc"
            fi
        done
    fi
fi
export SUMO_HOME

# 3. Виртуальное окружение
echo "Создание виртуального окружения..."
if [[ -d venv && ! -f venv/bin/activate ]]; then
    echo "Удаление неполного venv/..."
    rm -rf venv
fi
python3 -m venv venv

if [[ ! -f venv/bin/activate ]]; then
    echo "Ошибка: venv не создан. На Debian/Ubuntu: sudo apt install python3-venv" >&2
    exit 1
fi

# shellcheck source=/dev/null
source venv/bin/activate

# 4. Python-зависимости (только внутри venv — без PEP 668)
echo "Установка Python-библиотек..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Настройка завершена!"
echo "  source venv/bin/activate"
echo "  export SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}"
