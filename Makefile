.PHONY: install install-dev help

help:
	@echo "Available targets:"
	@echo "  make install      - Install cruijff_kit with all dependencies (torchtune nightly)"
	@echo "  make install-dev  - Install with dev dependencies (pytest, pytest-cov) + reminder for gh"

install:
	@echo "Installing cruijff_kit with dependencies..."
	pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126
	@echo "Upgrading torchtune to nightly build (for val_loss tracking)..."
	pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	@echo "Installation complete!"

install-dev:
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
