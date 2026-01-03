import re

# Read the current settings.py
with open('src/config/settings.py', 'r') as f:
    content = f.read()

# Find the CORS section and add API configuration after it
# Look for the line after allowed_headers field
pattern = r'(allowed_headers: List\[str\] = Field\(\s*default=\["\*"\],\s*description="Allowed HTTP headers"\s*\)\s*)'

# Add API configuration after CORS section
replacement = r'''\1
    
    # API Configuration
    api_prefix: str = Field(default="/api/v1", description="API URL prefix")
    api_title: str = Field(default="AFIE API", description="API documentation title")
    api_description: str = Field(default="Adaptive Farm Intelligence Engine API", description="API description")
    api_version: str = Field(default="0.1.0", description="API version")'''

# Make the replacement
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('src/config/settings.py', 'w') as f:
    f.write(new_content)

print("Added API configuration to settings.py")
