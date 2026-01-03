import re

# Read the file
with open('src/main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the right place to insert Redis variables
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    
    # After the docstring and initial variables, add Redis variables
    if 'startup_errors = []' in line:
        new_lines.append('\n')
        new_lines.append('    # Redis variables (optional)\n')
        new_lines.append('    redis_client = None\n')
        new_lines.append('    redis_available = False\n')
        new_lines.append('\n')
    
    # Remove the duplicate definitions inside the except block
    elif 'redis_client = None' in line and 'redis_available = False' in lines[i+1]:
        # Skip adding these lines - they're duplicates
        # Also skip the next line
        pass
    elif 'redis_available = False' in line and 'try:' in lines[i-1]:
        # Skip this line too
        pass

# Write back
with open('src/main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed Redis variable scope")
