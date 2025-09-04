#!/usr/bin/env python3
# check_environment.py - Compare local vs VPS environment

import subprocess
import sys
import os
import platform
import psutil

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
    except:
        return "Command failed"

def check_environment():
    """Comprehensive environment check"""
    
    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # 1. System Info
    print("\n### SYSTEM INFO ###")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # 2. Memory
    print("\n### MEMORY ###")
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"Used RAM: {mem.percent}%")
    
    swap = psutil.swap_memory()
    print(f"Swap Total: {swap.total / (1024**3):.2f} GB")
    print(f"Swap Used: {swap.percent}%")
    
    # 3. CPU
    print("\n### CPU ###")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # 4. Disk
    print("\n### DISK ###")
    disk = psutil.disk_usage('/')
    print(f"Disk Total: {disk.total / (1024**3):.2f} GB")
    print(f"Disk Free: {disk.free / (1024**3):.2f} GB")
    print(f"Disk Used: {disk.percent}%")
    
    # 5. Python Packages
    print("\n### PYTHON PACKAGES ###")
    packages = [
        'streamlit',
        'pytesseract',
        'pdf2image',
        'PyPDF2',
        'Pillow',
        'anthropic',
        'pandas',
        'numpy',
        'scikit-learn',
        'python-docx',
        'easyocr'
    ]
    
    for package in packages:
        try:
            module = __import__(package.replace('-', '_'))
            version = getattr(module, '__version__', 'installed')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")
    
    # 6. System Libraries
    print("\n### SYSTEM LIBRARIES ###")
    
    # Check Tesseract
    tesseract = run_command("tesseract --version | head -n 1")
    print(f"Tesseract: {tesseract}")
    
    # Check Tesseract languages
    langs = run_command("tesseract --list-langs 2>/dev/null | tail -n +2")
    print(f"Tesseract Languages: {langs[:100]}...")
    
    # Check poppler
    poppler = run_command("pdftoppm -v 2>&1 | head -n 1")
    print(f"Poppler: {poppler}")
    
    # Check ImageMagick
    imagemagick = run_command("convert -version | head -n 1")
    print(f"ImageMagick: {imagemagick}")
    
    # Check Ghostscript
    gs = run_command("gs --version")
    print(f"Ghostscript: {gs}")
    
    # 7. Environment Variables
    print("\n### ENVIRONMENT VARIABLES ###")
    important_vars = [
        'TESSDATA_PREFIX',
        'OMP_THREAD_LIMIT',
        'PYTHONPATH',
        'PATH',
        'LANG',
        'LC_ALL',
        'ANTHROPIC_API_KEY'
    ]
    
    for var in important_vars:
        value = os.environ.get(var, 'NOT SET')
        if var == 'ANTHROPIC_API_KEY' and value != 'NOT SET':
            value = value[:10] + '***'  # Hide API key
        print(f"{var}: {value}")
    
    # 8. Process Limits
    print("\n### PROCESS LIMITS ###")
    ulimit = run_command("ulimit -a")
    print(ulimit)
    
    # 9. Temp Directory
    print("\n### TEMP DIRECTORY ###")
    temp_dir = run_command("df -h /tmp | tail -n 1")
    print(f"/tmp usage: {temp_dir}")
    
    # Check write permission
    test_file = '/tmp/test_write.txt'
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✓ Can write to /tmp")
    except:
        print("✗ Cannot write to /tmp")
    
    # 10. Network
    print("\n### NETWORK ###")
    print(f"Hostname: {platform.node()}")
    
    # Check localhost connectivity
    localhost = run_command("curl -s -o /dev/null -w '%{http_code}' http://localhost:8501")
    print(f"Streamlit (localhost:8501): {localhost}")
    
    # 11. Systemd Service Status (if on VPS)
    print("\n### SERVICE STATUS ###")
    if os.path.exists('/etc/systemd/system/streamlit-app.service'):
        status = run_command("systemctl status streamlit-app --no-pager | head -n 20")
        print(status)
    else:
        print("Not running as systemd service")
    
    print("\n" + "=" * 60)
    print("END OF DIAGNOSTIC REPORT")
    print("=" * 60)

if __name__ == "__main__":
    check_environment()