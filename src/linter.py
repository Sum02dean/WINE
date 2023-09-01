import sys
from pylint import lint

THRESHOLD = 8

# List of files to lint
files_to_lint = [
    "preliminary_model.py",
    "simple_torch_nn.py",
    "random_forest.py",
   "support_vector_classifier.py"
]

# Additional arguments for pylint
pylint_args = [
    "--disable=E1101",  # Disable error E1101 (no-member)
    "--ignored-modules=torch" # Ignore imports from the 'torc' module
]

run = lint.Run(files_to_lint + pylint_args, do_exit=False)
score = run.linter.stats.global_note

if score < THRESHOLD:
    print("Linter failed: Score < threshold value")
    sys.exit(1)
sys.exit(0)