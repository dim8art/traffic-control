#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MIN_SUMO_VERSION="1.18.0"

find_sumo_bin() {
    local candidate
    for candidate in "${SUMO_BIN:-}" sumo "${SUMO_HOME:-}/bin/sumo"; do
        [[ -n "$candidate" && -x "$candidate" ]] || continue
        printf '%s' "$candidate"
        return 0
    done
    return 1
}

sumo_version() {
    local sumo_bin line ver
    sumo_bin="$(find_sumo_bin)" || return 1
    line="$("$sumo_bin" --version 2>&1 | head -1)"
    # Eclipse SUMO sumo Version 1.26.0  /  SUMO Version 1.26.0  /  1.26.0
    ver="$(printf '%s\n' "$line" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    if [[ -z "$ver" ]]; then
        ver="$(printf '%s\n' "$line" | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    fi
    [[ -n "$ver" ]] || return 1
    printf '%s' "$ver"
}

require_sumo_min() {
    local ver="$1"
    local sumo_bin line
    if [[ -z "$ver" ]]; then
        echo "Ошибка: не удалось определить версию SUMO." >&2
        if sumo_bin="$(find_sumo_bin)"; then
            line="$("$sumo_bin" --version 2>&1 | head -3)"
            echo "  sumo: ${sumo_bin}" >&2
            echo "  вывод: ${line:-<пусто>}" >&2
        else
            echo "  sumo не найден в PATH и в \${SUMO_HOME}/bin/sumo" >&2
            echo "  Ubuntu: sudo add-apt-repository -y ppa:sumo/stable" >&2
            echo "          sudo apt install sumo sumo-tools sumo-doc" >&2
        fi
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

export PATH="${SUMO_HOME}/bin:${PATH}"

INSTALLED_SUMO_VER="$(sumo_version)" || INSTALLED_SUMO_VER=""
require_sumo_min "$INSTALLED_SUMO_VER"
echo "SUMO ${INSTALLED_SUMO_VER} (${SUMO_HOME:-не задан}, $(find_sumo_bin))"

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
