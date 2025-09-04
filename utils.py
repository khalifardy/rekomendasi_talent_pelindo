import re

def sanitize_filename(filename):
    """Remove forbidden characters for Files API"""
    # Extract just the filename from path if it's a path
    import os
    filename = os.path.basename(filename)
    
    # Remove/replace forbidden characters
    forbidden_chars = r'[<>:"|?*\\/\x00-\x1f]'
    clean_name = re.sub(forbidden_chars, '_', filename)
    
    # Ensure not empty and within 1-255 chars
    if not clean_name:
        clean_name = 'document.pdf'
    
    if len(clean_name) > 255:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:250] + ext
    
    return clean_name