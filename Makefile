VENV_NAME = venv
VENV_BIN = $(VENV_NAME)/bin
PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/pip

REQUIREMENTS = requirements.txt

.PHONY: all venv install clean start

all: venv install

venv:
	@echo "Создание виртуальной среды..."
	python3 -m venv $(VENV_NAME)
	@echo "Виртуальная среда создана в папке $(VENV_NAME)"

install: venv
	@echo "Установка зависимостей..."
	$(PIP) install --upgrade pip
	if [ -f $(REQUIREMENTS) ]; then \
		$(PIP) install -r $(REQUIREMENTS); \
	else \
		echo "Файл $(REQUIREMENTS) не найден, устанавливаю основные библиотеки..."; \
		$(PIP) install numpy pandas matplotlib requests; \
	fi
	@echo "Зависимости установлены"

gptvers: install
	@echo "Запуск gpt.py..."
	$(PYTHON) gptversion.py

start: install
	@echo "Запуск main.py..."
	$(PYTHON) main.py

clean:
	@echo "Очистка виртуальной среды..."
	rm -rf $(VENV_NAME)
	@echo "Виртуальная среда удалена"

help:
	@echo "Доступные команды:"
	@echo "  make venv     - Создать виртуальную среду"
	@echo "  make install  - Установить зависимости"
	@echo "  make start    - Запустить main.py"
	@echo "  make clean    - Удалить виртуальную среду"
	@echo "  make all      - Создать среду и установить зависимости"