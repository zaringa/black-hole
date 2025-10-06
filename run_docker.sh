#!/bin/bash

if ! command -v docker &> /dev/null; then
    echo "Ошибка: Docker не найден. Убедитесь что Docker установлен и добавлен в PATH" >&2
    echo "Можно установить Docker: https://docs.docker.com/get-docker/" >&2
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    echo "Ошибка: Dockerfile не найден в корне проекта" >&2
    exit 1
fi

echo "Очистка старого образа..."
docker rmi -f decimal-tests &> /dev/null

echo "Сборка Docker образа..."
docker build 
