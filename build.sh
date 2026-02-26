#!/bin/bash
# 1. Обновление системы и установка системных зависимостей для SUMO
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Установка системных зависимостей (Linux)..."
    sudo apt-get update
    sudo apt-get install -y sumo sumo-tools sumo-doc python3-pip
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Установка системных зависимостей (macOS)..."
    brew install sumo
fi

# 2. Проверка переменной окружения SUMO_HOME (необходима для работы библиотек)
if [ -z "$SUMO_HOME" ]; then
    echo "Настройка SUMO_HOME..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        export SUMO_HOME="/usr/share/sumo"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        export SUMO_HOME="/usr/local/opt/sumo/share/sumo"
    fi
    echo "export SUMO_HOME=$SUMO_HOME" >> ~/.bashrc
    echo "export SUMO_HOME=$SUMO_HOME" >> ~/.zshrc
fi

# 3. Создание виртуального окружения Python
echo "Создание виртуального окружения..."
python3 -m venv venv
source venv/bin/activate

# 4. Обновление pip и установка зависимостей
echo "Установка Python-библиотек..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Настройка завершена!"
echo "Для активации окружения используйте: source venv/bin/activate"