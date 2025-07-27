# Find and replace the evaluation section in the file
import fileinput
import sys

with open('deepearth_crossmodal_training_optimized.py', 'r') as f:
    content = f.read()

# Fix the evaluation call
old_line = "outputs = model(vision, None, mask_language=False, mask_prob=0)"
new_line = "outputs = model(vision, batch['language_embedding'].to(device), mask_language=False, mask_prob=0)"

content = content.replace(old_line, new_line)

with open('deepearth_crossmodal_training_optimized.py', 'w') as f:
    f.write(content)

print("✓ Fixed evaluation code")
