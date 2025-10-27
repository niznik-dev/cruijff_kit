#!/usr/bin/env python3
"""
Generate interactive HTML visualizations from Inspect AI evaluation results.

This script uses the inspect-viz library to create custom and pre-built
visualizations of experiment evaluation data. It's designed to work with
both new (eval/logs/) and legacy (logs/) directory structures.

Usage:
    # From experiment directory
    python tools/inspect/run_inspect_viz.py

    # Specify experiment directory
    python tools/inspect/run_inspect_viz.py --experiment_dir /path/to/experiment

Example:
    cd /scratch/gpfs/MSALGANIK/mjs3/cap_7L_llama32_lora_comparison_2025-10-18
    python ~/cruijff_kit/tools/inspect/run_inspect_viz.py
"""

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger


class InspectVizRunner:
    """
    Manages the generation of visualizations from Inspect AI evaluation results.

    This class handles:
    - Experiment directory discovery and validation
    - Flexible .eval file discovery (multiple directory structures)
    - Logging and error handling
    - Visualization generation workflow
    """

    def __init__(self, experiment_dir: Optional[Path] = None, log_file: str = "run-inspect-viz.log"):
        """
        Initialize the visualization runner.

        Args:
            experiment_dir: Path to experiment directory. If None, uses current directory.
            log_file: Name of log file to create in experiment directory.
        """
        # Determine experiment directory
        if experiment_dir is None:
            self.experiment_dir = Path.cwd()
        else:
            self.experiment_dir = Path(experiment_dir).resolve()

        # Set up logging - console only initially, add file later if dir exists
        self.logger = setup_logger(
            __name__,
            console=True
        )
        self.log_file_name = log_file

        # Initialize state
        self.eval_files: List[Path] = []
        self.design_doc: Optional[Path] = None

    def log_action(self, action: str, details: str, result: str):
        """
        Log an action in the standardized format.

        Args:
            action: Action name (e.g., "DISCOVER_EXPERIMENT")
            details: Details about what was attempted
            result: Outcome of the action
        """
        self.logger.info(f"{action}: {details}")
        self.logger.info(f"Details: {details}")
        self.logger.info(f"Result: {result}\n")

    def discover_experiment(self) -> bool:
        """
        Validate that the experiment directory exists and is accessible.

        Returns:
            True if experiment directory is valid, False otherwise.
        """
        self.logger.info("="*60)
        self.logger.info(f"Starting run-inspect-viz at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*60)

        if not self.experiment_dir.exists():
            self.logger.error(f"DISCOVER_EXPERIMENT: ERROR")
            self.logger.error(f"Directory does not exist: {self.experiment_dir}")
            self.logger.error("\nPlease check:")
            self.logger.error("  - Path is correct")
            self.logger.error("  - Directory has been created")
            self.logger.error("  - You have access permissions")
            return False

        if not self.experiment_dir.is_dir():
            self.logger.error(f"DISCOVER_EXPERIMENT: ERROR")
            self.logger.error(f"Path exists but is not a directory: {self.experiment_dir}")
            return False

        # Now that we know the directory exists, set up file logging
        log_path = self.experiment_dir / self.log_file_name
        # Reconfigure logger to add file handler
        self.logger = setup_logger(
            __name__,
            log_file=str(log_path),
            console=True
        )

        self.log_action(
            "DISCOVER_EXPERIMENT",
            f"Found experiment at: {self.experiment_dir}",
            "Experiment directory located successfully"
        )
        return True

    def discover_eval_files(self) -> bool:
        """
        Find all .eval files in the experiment directory.

        Searches in both:
        - New structure: {run_dir}/eval/logs/*.eval
        - Legacy structure: {run_dir}/logs/*.eval

        Returns:
            True if at least one .eval file found, False otherwise.
        """
        self.log_action(
            "DISCOVER_EVALS",
            "Scanning for .eval files in both new (eval/logs/) and legacy (logs/) locations",
            "Starting search..."
        )

        # Search patterns for both directory structures
        patterns = [
            str(self.experiment_dir / "**/eval/logs/*.eval"),  # New structure
            str(self.experiment_dir / "**/logs/*.eval"),        # Legacy structure
            str(self.experiment_dir / "*.eval"),                # Root level (edge case)
        ]

        eval_files = set()  # Use set to avoid duplicates

        for pattern in patterns:
            found = glob.glob(pattern, recursive=True)
            eval_files.update(found)

        # Convert to sorted list of Path objects
        self.eval_files = sorted([Path(f) for f in eval_files])

        if not self.eval_files:
            self.log_action(
                "DISCOVER_EVALS",
                f"Searched patterns: {patterns}",
                "ERROR: No .eval files found"
            )
            self.logger.error("\n❌ No evaluation files found!")
            self.logger.error("Possible causes:")
            self.logger.error("  1. Evaluations have not been run yet")
            self.logger.error("  2. .eval files are in an unexpected location")
            self.logger.error("  3. Wrong experiment directory")
            self.logger.error("\nSuggestions:")
            self.logger.error("  - Run evaluations first with the run-inspect skill")
            self.logger.error("  - Check that you're in the correct experiment directory")
            self.logger.error("  - Verify .eval files exist with: find . -name '*.eval'")
            return False

        # Log details about found files
        file_locations = {}
        for eval_file in self.eval_files:
            # Categorize by directory structure
            if "/eval/logs/" in str(eval_file):
                structure = "new (eval/logs/)"
            elif "/logs/" in str(eval_file):
                structure = "legacy (logs/)"
            else:
                structure = "root"

            file_locations[structure] = file_locations.get(structure, 0) + 1

        location_summary = ", ".join([f"{count} in {structure}" for structure, count in file_locations.items()])

        self.log_action(
            "DISCOVER_EVALS",
            f"Found {len(self.eval_files)} .eval files",
            f"Success: {location_summary}"
        )

        # Log individual files for debugging
        self.logger.info("Evaluation files found:")
        for eval_file in self.eval_files:
            # Show path relative to experiment dir for readability
            rel_path = eval_file.relative_to(self.experiment_dir)
            self.logger.info(f"  - {rel_path}")
        self.logger.info("")

        return True

    def run(self) -> bool:
        """
        Execute the complete visualization workflow.

        Returns:
            True if successful, False if any critical step fails.
        """
        # Step 1: Discover experiment
        if not self.discover_experiment():
            return False

        # Step 2: Discover .eval files
        if not self.discover_eval_files():
            return False

        # Success summary for this chunk
        self.logger.info("="*60)
        self.logger.info("✓ Chunk 2 Complete: Experiment Discovery & Validation")
        self.logger.info("="*60)
        self.logger.info(f"Experiment: {self.experiment_dir}")
        self.logger.info(f"Evaluation files found: {len(self.eval_files)}")
        self.logger.info("Next: Load evaluation data and parse experimental design")
        self.logger.info("="*60)

        return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate interactive visualizations from Inspect AI evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in current directory (experiment directory)
  python run_inspect_viz.py

  # Specify experiment directory
  python run_inspect_viz.py --experiment_dir /scratch/gpfs/MSALGANIK/mjs3/my_experiment

  # Custom log file name
  python run_inspect_viz.py --log_file my_viz_log.log
        """
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to experiment directory (default: current directory)"
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="run-inspect-viz.log",
        help="Name of log file to create (default: run-inspect-viz.log)"
    )

    args = parser.parse_args()

    # Create runner and execute
    runner = InspectVizRunner(
        experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        log_file=args.log_file
    )

    success = runner.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
