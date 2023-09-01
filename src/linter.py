# Create a pylint script to test preliminary_model.py
import sys 
from pylint import lint  

THRESHOLD = 8

run = lint.Run([
    "preliminary_model.py",
    "simple_torch_nn.py",
    "random_forest.py",
    "support_vector_classifier.py"], do_exit=False)
score = run.linter.stats.global_note

if score < THRESHOLD:
    print("Linter failed: Score < threshold value")
    sys.exit(1)
sys.exit(0)
