# Save as /d/agrogram-afie/src/utils/encoding_fix.py
import sys
import io

def fix_windows_encoding():
    """Fix Windows console encoding issues."""
    if sys.platform == "win32":
        # Set UTF-8 encoding for Windows console
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        return True
    return False

# Apply fix immediately
fix_windows_encoding()