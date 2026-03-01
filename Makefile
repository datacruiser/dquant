.PHONY: help install test clean lint format docs run

help:
	@echo "DQuant Makefile"
	@echo ""
	@echo "使用: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install     安装依赖"
	@echo "  test        运行测试"
	@echo "  clean       清理缓存"
	@echo "  lint        代码检查"
	@echo "  format      格式化代码"
	@echo "  docs        生成文档"
	@echo "  run         运行快速开始"
	@echo "  example     运行示例"

install:
	pip install -e .
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-full.txt

test:
	python tests/test_basic.py
	python tests/test_factors.py

test-verbose:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

lint:
	flake8 dquant/ --max-line-length=100 --exclude=__pycache__

format:
	black dquant/ tests/ examples/
	isort dquant/ tests/ examples/

docs:
	cd docs && make html

run:
	python quickstart.py

example:
	python examples/simple_backtest.py

benchmark:
	python -m timeit -s "from dquant import get_factor" "get_factor('momentum')"
