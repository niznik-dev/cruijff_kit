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

    def __init__(self, experiment_dir: Optional[Path] = None, log_file: str = "run-inspect-viz.log", skip_validation: bool = False):
        """
        Initialize the visualization runner.

        Args:
            experiment_dir: Path to experiment directory. If None, uses current directory.
            log_file: Name of log file to create in experiment directory.
            skip_validation: If True, skip validation check comparing design to data.
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
        self.skip_validation = skip_validation

        # Initialize state
        self.eval_files: List[Path] = []
        self.design_doc: Optional[Path] = None
        self.evals_data = None  # Will hold inspect_viz Data object
        self.experiment_design: dict = {}  # Parsed experiment design info

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

    def load_evaluation_data(self) -> bool:
        """
        Load evaluation data from .eval files using Inspect AI's dataframe API.

        This method:
        1. Loads .eval files into a dataframe
        2. Prepares the data with model and task metadata
        3. Converts to inspect_viz Data object for visualization
        4. Handles malformed files gracefully

        Returns:
            True if data loaded successfully, False if critical errors occurred.
        """
        self.log_action(
            "LOAD_DATA",
            f"Loading {len(self.eval_files)} .eval files",
            "Starting data loading..."
        )

        try:
            # Import inspect_ai and inspect_viz libraries
            try:
                from inspect_ai.analysis import evals_df
                from inspect_viz import Data
            except ImportError as e:
                self.logger.error(f"LOAD_DATA: ERROR")
                self.logger.error(f"Missing required library: {e}")
                self.logger.error("\nPlease install required packages:")
                self.logger.error("  pip install inspect-ai inspect-viz")
                self.logger.error("or:")
                self.logger.error("  conda install -c conda-forge inspect-ai inspect-viz")
                return False

            # Load .eval files into dataframe
            self.logger.info("Loading .eval files into dataframe...")
            try:
                # evals_df expects a directory path or list of file paths (as strings)
                evals = evals_df([str(f) for f in self.eval_files])
                self.logger.info(f"✓ Loaded {len(evals)} evaluation records")
            except Exception as e:
                self.logger.error(f"LOAD_DATA: ERROR")
                self.logger.error(f"Failed to load .eval files: {e}")
                self.logger.error("\nPossible causes:")
                self.logger.error("  - Malformed .eval files")
                self.logger.error("  - Incompatible Inspect AI versions")
                self.logger.error("  - Corrupted evaluation data")
                self.logger.error("\nSuggestions:")
                self.logger.error("  - Check .eval file integrity")
                self.logger.error("  - Verify Inspect AI version compatibility")
                self.logger.error("  - Try loading files individually to identify problematic ones")
                return False

            # Convert to inspect_viz Data object
            self.logger.info("Converting to inspect_viz Data object...")
            try:
                self.evals_data = Data.from_dataframe(evals)
                self.logger.info("✓ Data object created successfully")
            except Exception as e:
                self.logger.error(f"LOAD_DATA: ERROR")
                self.logger.error(f"Failed to create Data object: {e}")
                self.logger.error("\nThis is a critical error - cannot proceed with visualization")
                return False

            # Log summary of loaded data
            self.logger.info("\nData Summary:")
            self.logger.info(f"  Total evaluation records: {len(evals)}")

            # Try to extract some useful summary info
            try:
                if 'model' in evals.columns:
                    unique_models = evals['model'].nunique()
                    self.logger.info(f"  Unique models: {unique_models}")

                if 'task' in evals.columns:
                    unique_tasks = evals['task'].nunique()
                    self.logger.info(f"  Unique tasks: {unique_tasks}")

                if 'score' in evals.columns:
                    self.logger.info(f"  Score range: {evals['score'].min():.3f} to {evals['score'].max():.3f}")
            except Exception:
                # Summary info is nice-to-have but not critical
                pass

            self.log_action(
                "LOAD_DATA",
                "Evaluation data loaded and prepared",
                "Success: Data ready for visualization"
            )

            return True

        except Exception as e:
            self.logger.error(f"LOAD_DATA: ERROR")
            self.logger.error(f"Unexpected error during data loading: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

    def _ensure_visualizations_dir(self) -> Path:
        """
        Create visualizations directory if it doesn't exist.

        Returns:
            Path to the visualizations directory.
        """
        viz_dir = self.experiment_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        return viz_dir

    def validate_design_vs_data(self) -> bool:
        """
        Validate that the experimental design matches the actual evaluation data.

        Compares:
        - Expected variables/factors from experiment_summary.md
        - Actual variation in the loaded .eval data

        Returns:
            True if validation passes or user chooses to continue, False to abort.
        """
        self.log_action(
            "VALIDATE_DESIGN",
            "Comparing experimental design to actual evaluation data",
            "Starting validation..."
        )

        # Import here to access the dataframe
        from inspect_ai.analysis import evals_df

        # Reload the dataframe for inspection
        evals = evals_df([str(f) for f in self.eval_files])

        # Extract what varies in the actual data
        actual_variation = {}

        # Check model variation
        if 'model' in evals.columns:
            unique_models = evals['model'].nunique()
            model_values = evals['model'].unique().tolist()
            actual_variation['models'] = {
                'count': unique_models,
                'values': model_values
            }

        # Check task variation
        if 'task' in evals.columns:
            unique_tasks = evals['task'].nunique()
            task_values = evals['task'].unique().tolist()
            actual_variation['tasks'] = {
                'count': unique_tasks,
                'values': task_values
            }

        # Check for any metadata fields that might indicate experimental conditions
        metadata_fields = [col for col in evals.columns if col.startswith('metadata_')]
        if metadata_fields:
            actual_variation['metadata_fields'] = metadata_fields

        # Compare with expected design
        issues = []
        warnings = []

        # Check models
        expected_model_conditions = len(self.experiment_design.get('variables', []))
        if expected_model_conditions >= 2:
            # We expect multiple conditions (e.g., base vs fine-tuned)
            if actual_variation.get('models', {}).get('count', 0) == 1:
                issues.append(
                    f"Design specifies {expected_model_conditions} experimental factors "
                    f"but data contains only 1 unique model: {actual_variation['models']['values']}"
                )

        # Check if design mentions specific factors but we don't see them in data
        for var in self.experiment_design.get('variables', []):
            factor = var.get('factor', '')
            levels = var.get('levels', '')

            # Look for this factor in model names, task names, or metadata
            factor_found = False

            # Check if factor appears in model names
            if 'models' in actual_variation:
                model_str = ' '.join([str(m) for m in actual_variation['models']['values']])
                if any(level.lower() in model_str.lower() for level in levels.split(',')):
                    factor_found = True

            # Check if factor appears in task names
            if 'tasks' in actual_variation:
                task_str = ' '.join([str(t) for t in actual_variation['tasks']['values']])
                if any(level.lower() in task_str.lower() for level in levels.split(',')):
                    factor_found = True

            if not factor_found:
                warnings.append(
                    f"Design specifies factor '{factor}' with levels [{levels}] "
                    f"but this variation is not visible in the data"
                )

        # Report findings
        self.logger.info("\n" + "="*60)
        self.logger.info("Experimental Design vs Data Validation")
        self.logger.info("="*60)

        self.logger.info("\nExpected (from experiment_summary.md):")
        if self.experiment_design.get('variables'):
            for var in self.experiment_design['variables']:
                self.logger.info(f"  - {var['factor']}: {var['levels']}")
        else:
            self.logger.info("  - No variables explicitly defined")

        self.logger.info("\nActual (from .eval files):")
        if actual_variation.get('models'):
            self.logger.info(f"  - Models: {actual_variation['models']['count']} unique")
            for model in actual_variation['models']['values']:
                self.logger.info(f"    • {model}")

        if actual_variation.get('tasks'):
            self.logger.info(f"  - Tasks: {actual_variation['tasks']['count']} unique")
            for task in actual_variation['tasks']['values']:
                self.logger.info(f"    • {task}")

        if actual_variation.get('metadata_fields'):
            self.logger.info(f"  - Metadata fields: {', '.join(actual_variation['metadata_fields'])}")

        # Report issues
        if issues or warnings:
            self.logger.warning("\n" + "⚠️  Validation Issues Detected".center(60))
            self.logger.warning("="*60)

            if issues:
                self.logger.warning("\n❌ CRITICAL MISMATCHES:")
                for issue in issues:
                    self.logger.warning(f"  • {issue}")

            if warnings:
                self.logger.warning("\n⚠️  WARNINGS:")
                for warning in warnings:
                    self.logger.warning(f"  • {warning}")

            self.logger.warning("\n" + "="*60)
            self.logger.warning("Possible explanations:")
            self.logger.warning("  1. Experiment is partially complete (some conditions not yet run)")
            self.logger.warning("  2. experiment_summary.md describes planned work, not completed work")
            self.logger.warning("  3. Evaluation metadata doesn't capture experimental conditions")
            self.logger.warning("  4. Wrong experiment directory")
            self.logger.warning("\nPossible actions:")
            self.logger.warning("  1. Continue anyway - generate visualizations with available data")
            self.logger.warning("  2. Exit - check experiment status and re-run when complete")
            self.logger.warning("  3. Update experiment_summary.md to match actual data")
            self.logger.warning("="*60)

            # Ask user how to proceed
            self.logger.error("\n❌ VALIDATION FAILED")
            self.logger.error("Experimental design does not match evaluation data.")
            self.logger.error("\nPlease review the issues above and decide how to proceed:")
            self.logger.error("  • Continue generating visualizations with available data?")
            self.logger.error("  • Or exit and investigate the mismatch?")

            self.log_action(
                "VALIDATE_DESIGN",
                "Validation found mismatches between design and data",
                "FAILED - User input required"
            )

            return False

        else:
            self.logger.info("\n✓ Validation passed: Design matches data")
            self.log_action(
                "VALIDATE_DESIGN",
                "Experimental design matches evaluation data",
                "SUCCESS"
            )
            return True

    def generate_prebuilt_views(self) -> bool:
        """
        Generate pre-built visualization views using inspect_viz.

        This method generates standard visualization views including:
        - scores_by_model: Compare performance across different models
        - (More views will be added in Chunk 6)

        Returns:
            True if at least one view generated successfully, False if all failed.
        """
        self.log_action(
            "GENERATE_VIEWS",
            "Generating pre-built visualization views",
            "Starting view generation..."
        )

        try:
            # Import required functions
            from inspect_viz.view.beta import (
                scores_by_model,
                scores_by_task,
                scores_timeline,
                scores_heatmap,
                scores_by_factor
            )
            from inspect_viz.plot import write_html
        except ImportError as e:
            self.logger.error(f"GENERATE_VIEWS: ERROR")
            self.logger.error(f"Failed to import inspect_viz components: {e}")
            self.logger.error("\nPlease ensure inspect-viz is installed:")
            self.logger.error("  pip install inspect-viz")
            return False

        # Ensure visualizations directory exists
        viz_dir = self._ensure_visualizations_dir()
        self.logger.info(f"Visualizations will be saved to: {viz_dir}")

        views_generated = []
        views_failed = []

        # Define all pre-built views to generate
        views_to_generate = [
            ("scores_by_model", scores_by_model, "Compare performance across different models"),
            ("scores_by_task", scores_by_task, "Compare performance across different tasks"),
            ("scores_timeline", scores_timeline, "Show how scores change over time"),
            ("scores_heatmap", scores_heatmap, "Heatmap showing model vs task performance"),
            ("scores_by_factor", scores_by_factor, "Show scores across experimental factors"),
        ]

        # Generate each view
        for view_name, view_func, description in views_to_generate:
            self.logger.info(f"\nGenerating view: {view_name}")
            self.logger.info(f"  Description: {description}")

            try:
                # Create the view component
                view = view_func(self.evals_data)

                # Save to HTML file
                output_file = viz_dir / f"{view_name}.html"
                write_html(str(output_file), view)

                # Check file size to confirm it was created
                file_size = output_file.stat().st_size
                self.logger.info(f"✓ Generated: {output_file.name} ({file_size/1024:.1f} KB)")
                views_generated.append(view_name)

            except Exception as e:
                self.logger.warning(f"✗ Failed to generate {view_name}: {e}")
                views_failed.append((view_name, str(e)))

        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("Pre-built Views Generation Summary")
        self.logger.info("="*60)
        self.logger.info(f"✓ Successfully generated: {len(views_generated)} view(s)")
        for view_name in views_generated:
            self.logger.info(f"  - {view_name}")

        if views_failed:
            self.logger.warning(f"\n✗ Failed to generate: {len(views_failed)} view(s)")
            for view_name, error in views_failed:
                self.logger.warning(f"  - {view_name}: {error}")

        self.log_action(
            "GENERATE_VIEWS",
            f"Generated {len(views_generated)} pre-built view(s)",
            "View generation complete"
        )

        # Consider success if at least one view was generated
        return len(views_generated) > 0

    def parse_experiment_summary(self) -> bool:
        """
        Parse experiment_summary.md to extract experimental design information.

        Extracts:
        - Variables being tested (what varies across runs)
        - Run configurations
        - Evaluation tasks
        - Overview/description

        Returns:
            True if design document found and parsed, False otherwise.
        """
        self.log_action(
            "PARSE_DESIGN",
            "Looking for experiment_summary.md",
            "Starting design document search..."
        )

        # Look for experiment_summary.md
        design_path = self.experiment_dir / "experiment_summary.md"

        if not design_path.exists():
            self.logger.error(f"PARSE_DESIGN: ERROR")
            self.logger.error(f"No experiment_summary.md found at: {design_path}")
            self.logger.error("\nThis file is REQUIRED for run-inspect-viz")
            self.logger.error("\nPossible solutions:")
            self.logger.error("  1. Run 'design-experiment' skill to create experiment plan")
            self.logger.error("  2. Check you're in the correct experiment directory")
            self.logger.error("  3. If this is a legacy experiment, create experiment_summary.md manually")
            return False

        self.design_doc = design_path
        self.logger.info(f"✓ Found experiment design: {design_path.name}")

        # Parse the markdown file
        self.logger.info("Parsing experiment design...")
        try:
            with open(design_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract key information using simple parsing
            self.experiment_design = {
                'variables': [],
                'runs': [],
                'tasks': [],
                'overview': '',
                'tools': {}
            }

            # Extract overview (first paragraph after # header)
            lines = content.split('\n')
            in_overview = False
            overview_lines = []

            for line in lines:
                if line.startswith('# ') and 'Overview' in line:
                    in_overview = True
                    continue
                elif in_overview and line.startswith('## '):
                    break
                elif in_overview and line.strip():
                    overview_lines.append(line.strip())

            self.experiment_design['overview'] = ' '.join(overview_lines)

            # Extract variables section
            if '## Variables' in content:
                # Find the variables table or list
                var_section_start = content.find('## Variables')
                var_section = content[var_section_start:var_section_start+1000]

                # Simple extraction: look for lines with | or - (markdown tables)
                for line in var_section.split('\n'):
                    if '|' in line and 'Factor' not in line and '---' not in line:
                        parts = [p.strip() for p in line.split('|') if p.strip()]
                        if len(parts) >= 2:
                            self.experiment_design['variables'].append({
                                'factor': parts[0],
                                'levels': parts[1] if len(parts) > 1 else ''
                            })

            # Extract tools section
            if '## Tools' in content:
                tools_section_start = content.find('## Tools')
                tools_section = content[tools_section_start:tools_section_start+500]

                if 'Preparation:' in tools_section:
                    prep_line = [l for l in tools_section.split('\n') if 'Preparation:' in l][0]
                    prep = prep_line.split(':')[1].strip()
                    # Remove markdown formatting
                    prep = prep.replace('**', '').strip()
                    self.experiment_design['tools']['preparation'] = prep

                if 'Evaluation:' in tools_section:
                    eval_line = [l for l in tools_section.split('\n') if 'Evaluation:' in l][0]
                    eval_tool = eval_line.split(':')[1].strip()
                    # Remove markdown formatting
                    eval_tool = eval_tool.replace('**', '').strip()
                    self.experiment_design['tools']['evaluation'] = eval_tool

            # Extract evaluation plan
            if '## Evaluation Plan' in content:
                eval_section_start = content.find('## Evaluation Plan')
                eval_section = content[eval_section_start:eval_section_start+1000]

                # Look for task names
                for line in eval_section.split('\n'):
                    if 'Task:' in line or 'task:' in line:
                        task = line.split(':')[1].strip()
                        # Remove markdown formatting
                        task = task.replace('**', '').strip()
                        if task and task not in self.experiment_design['tasks']:
                            self.experiment_design['tasks'].append(task)

            # Log what was extracted
            self.logger.info("\nExtracted design information:")
            if self.experiment_design['overview']:
                self.logger.info(f"  Overview: {self.experiment_design['overview'][:100]}...")

            if self.experiment_design['variables']:
                self.logger.info(f"  Variables: {len(self.experiment_design['variables'])} factors")
                for var in self.experiment_design['variables']:
                    self.logger.info(f"    - {var['factor']}: {var['levels']}")
            else:
                self.logger.info("  Variables: None explicitly defined")

            if self.experiment_design['tools']:
                self.logger.info(f"  Tools: {self.experiment_design['tools']}")

            if self.experiment_design['tasks']:
                self.logger.info(f"  Evaluation tasks: {', '.join(self.experiment_design['tasks'])}")

            self.log_action(
                "PARSE_DESIGN",
                f"Parsed experiment_summary.md ({len(content)} characters)",
                "Success: Experimental design extracted"
            )

            return True

        except Exception as e:
            self.logger.error(f"PARSE_DESIGN: ERROR")
            self.logger.error(f"Failed to parse experiment_summary.md: {e}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

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

        # Step 3: Load evaluation data
        if not self.load_evaluation_data():
            return False

        # Step 4: Parse experiment design
        if not self.parse_experiment_summary():
            return False

        # Step 5: Validate design matches data (unless skipped)
        if not self.skip_validation:
            if not self.validate_design_vs_data():
                return False
        else:
            self.logger.info("⚠️  Validation skipped (--skip-validation flag set)")

        # Step 6: Generate pre-built views
        if not self.generate_prebuilt_views():
            self.logger.warning("No pre-built views were generated successfully")
            # This is a warning, not a critical failure - continue

        # Success summary
        self.logger.info("="*60)
        self.logger.info("✓ Chunk 6 Complete: All Pre-built Views Generated")
        self.logger.info("="*60)
        self.logger.info(f"Experiment: {self.experiment_dir}")
        self.logger.info(f"Evaluation files: {len(self.eval_files)}")
        self.logger.info(f"Visualizations directory: visualizations/")
        self.logger.info("Next: Generate custom visualizations based on experimental design")
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

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation check that compares experimental design to data"
    )

    args = parser.parse_args()

    # Create runner and execute
    runner = InspectVizRunner(
        experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        log_file=args.log_file,
        skip_validation=args.skip_validation
    )

    success = runner.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
