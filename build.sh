#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MIN_SUMO_VERSION="1.18.0"

sumo_version() {
    sumo --version 2>/dev/null | head -1 | sed -n 's/.*Version \([0-9][0-9.]*\).*/\1/p'
}

require_sumo_min() {
    local ver="$1"
    if [[ -z "$ver" ]]; then
        echo "Ошибка: не удалось определить версию SUMO (sumo --version)." >&2
        exit 1
    fi
    if ! printf '%s\n%s\n' "$MIN_SUMO_VERSION" "$ver" | sort -C -V; then
        echo "Ошибка: нужен SUMO >= ${MIN_SUMO_VERSION}, установлен ${ver}." >&2
        echo "  Ubuntu: sudo add-apt-repository -y ppa:sumo/stable && sudo apt install sumo sumo-tools sumo-doc" >&2
        exit 1
    fi
}

append_env_once() {
    local rc="$1"
    local line="$2"
    if [[ -f "$rc" ]] && ! grep -qF "$line" "$rc" 2>/dev/null; then
        echo "$line" >>"$rc"
    fi
}

# 1. Системные зависимости
if [[ "${OSTYPE:-}" == linux-gnu* ]]; then
    echo "Установка системных зависимостей (Linux)..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common curl
    echo "Подключение PPA sumo/stable (актуальный SUMO, не пакет из Ubuntu по умолчанию)..."
    sudo add-apt-repository -y ppa:sumo/stable
    sudo apt-get update
    sudo apt-get install -y \
        sumo sumo-tools sumo-doc \
        python3 python3-pip python3-venv \
        gdal-bin libgdal-dev libgeos-dev libproj-dev
elif [[ "${OSTYPE:-}" == darwin* ]]; then
    echo "Установка системных зависимостей (macOS)..."
    brew update
    brew install sumo
else
    echo "Предупреждение: неизвестная ОС (${OSTYPE:-}), пропуск apt/brew." >&2
fi

# 2. SUMO_HOME и PATH
if [[ -z "${SUMO_HOME:-}" ]]; then
    if [[ "${OSTYPE:-}" == linux-gnu* ]]; then
        export SUMO_HOME="/usr/share/sumo"
    elif [[ "${OSTYPE:-}" == darwin* ]]; then
        if [[ -d "/opt/homebrew/share/sumo" ]]; then
            export SUMO_HOME="/opt/homebrew/share/sumo"
        else
            export SUMO_HOME="/usr/local/opt/sumo/share/sumo"
        fi
    fi
fi

INSTALLED_SUMO_VER="$(sumo_version)"
require_sumo_min "$INSTALLED_SUMO_VER"
echo "SUMO ${INSTALLED_SUMO_VER} (${SUMO_HOME:-не задан})"

export PATH="${SUMO_HOME}/bin:${PATH}"

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    append_env_once "$rc" "export SUMO_HOME=${SUMO_HOME}"
    append_env_once "$rc" 'export PATH="${SUMO_HOME}/bin:${PATH}"'
done

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

# 4. Python: traci/sumolib той же версии, что бинарник SUMO
echo "Установка Python-библиотек..."
python -m pip install --upgrade pip
python -m pip install "traci==${INSTALLED_SUMO_VER}" "sumolib==${INSTALLED_SUMO_VER}"
python -m pip install -r requirements.txt

echo ""
echo "Проверка TraCI..."
python - <<PY
import traci
import sumolib
print(f"  traci {getattr(traci, '__version__', 'n/a')}  sumolib {sumolib.__version__}")
assert hasattr(traci.edge, "getLastStepPersonIDs"), "edge.getLastStepPersonIDs missing"
print("  edge.getLastStepPersonIDs — OK")
PY

echo ""
echo "Настройка завершена!"
echo "  source venv/bin/activate"
echo "  export SUMO_HOME=${SUMO_HOME}"
echo "  sumo --version"
