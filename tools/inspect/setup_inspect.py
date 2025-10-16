import os
import shutil
import sys
import argparse

# If possible, see if the input file comes from one of our test directories and if so grab the inspect.py from there
parser = argparse.ArgumentParser(description="Set up the inspect.py file for the current project.")
parser.add_argument("source_dir", type=str, required=True, help="Path to the directory containing inspect.py")
args = parser.parse_args()

source_file = os.path.join(args.source_dir, "inspect.py")
destination_file = "inspect.py"

if not os.path.isfile(source_file):
    print(f"Error: {source_file} does not exist.")
    sys.exit(1)

shutil.copy(source_file, destination_file)
print(f"Copied {source_file} to {destination_file}.")

# If we don't know where the input data came from, warn the user they'll have to supply their own inspect.py

# OK now we can parse the slurm script and make minimal changes
# A list of those would be: match nodes/ntasks/cpus-per-task/mem/time to the model output's slurm params (do we have those?)
# Allow account/partition/constraint to be changed
# Update conda env name if needed
# Update output dir
# Otherwise this one is much simpler than the finetune one. We could have this generate the inspect.py but that's a lot more work
# So for now we just copy the file and make minimal changes to the slurm script
# For now, just copy the template slurm script and tell the user to edit it

with open("templates/inspect_template.slurm", "r") as f:
    slurm_script = f.read()

# TODO - I'll work on this later

print("Ready to inspect with `sbatch inspect.slurm` !")