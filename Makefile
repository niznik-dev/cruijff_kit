.PHONY: install install-dev help

help:
	@echo "Available targets:"
	@echo "  make install      - Install cruijff_kit with all dependencies (torchtune nightly)"
	@echo "  make install-dev  - Install with dev dependencies (pytest, pytest-cov) + reminder for gh"

install:
	@if [ -z "$$VIRTUAL_ENV" ] && [ -z "$$CONDA_DEFAULT_ENV" ]; then \
		echo "❌ No Python environment detected!"; \
		echo ""; \
		echo "Create and activate an environment first:"; \
		echo "  Conda: conda create -n cruijff python=3.13 -y && conda activate cruijff"; \
		echo "  Venv:  python3.13 -m venv cruijff && source cruijff/bin/activate"; \
		echo ""; \
		exit 1; \
	elif [ "$$CONDA_DEFAULT_ENV" = "base" ]; then \
		echo "❌ Cannot install into conda base environment!"; \
		echo ""; \
		echo "Create and activate a dedicated environment:"; \
		echo "  conda create -n cruijff python=3.13 -y && conda activate cruijff"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Installing cruijff_kit with dependencies..."
	pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126
	@echo "Upgrading torchtune to nightly build (for val_loss tracking)..."
	pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	@echo "Installation complete!"

install-dev:
	@if [ -z "$$VIRTUAL_ENV" ] && [ -z "$$CONDA_DEFAULT_ENV" ]; then \
		echo "❌ No Python environment detected!"; \
		echo ""; \
		echo "Create and activate an environment first:"; \
		echo "  Conda: conda create -n cruijff python=3.13 -y && conda activate cruijff"; \
		echo "  Venv:  python3.13 -m venv cruijff && source cruijff/bin/activate"; \
		echo ""; \
		exit 1; \
	elif [ "$$CONDA_DEFAULT_ENV" = "base" ]; then \
		echo "❌ Cannot install into conda base environment!"; \
		echo ""; \
		echo "Create and activate a dedicated environment:"; \
		echo "  conda create -n cruijff python=3.13 -y && conda activate cruijff"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Installing cruijff_kit with dev dependencies..."
	pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
	@echo "Upgrading torchtune to nightly build (for val_loss tracking)..."
	pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	@echo ""
	@if [ -n "$$CONDA_DEFAULT_ENV" ]; then \
		echo "Conda environment detected - installing GitHub CLI..."; \
		conda install -c conda-forge gh -y; \
	else \
		echo "⚠️  Not using conda. Install GitHub CLI manually:"; \
		echo "    https://github.com/cli/cli#installation"; \
	fi
	@echo ""
	@echo "Installation complete!"
