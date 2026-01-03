import re

# Read the file
with open('src/core/farm_intelligence/farm_profiler.py', 'r') as f:
    lines = f.readlines()

# Find the line with farmer_id issue
for i, line in enumerate(lines):
    if "farmer_id=minimal_data.get('farmer_id'" in line:
        # Replace the problematic line
        lines[i] = '            farmer_id=uuid.UUID(minimal_data.get(\'farmer_id\') or minimal_data.get(\'farm_id\') or minimal_data.get(\'id\') or str(uuid.uuid4())),\n'
        break

# Write back
with open('src/core/farm_intelligence/farm_profiler.py', 'w') as f:
    f.writelines(lines)

print("Fixed UUID issue in farm_profiler.py")
