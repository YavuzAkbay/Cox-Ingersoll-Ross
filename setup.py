#!/usr/bin/env python3
"""
Setup script for Cox-Ingersoll-Ross Interest Rate Models

This script helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible!")
        return True

def setup_virtual_environment():
    """Set up virtual environment"""
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists!")
        return True
    
    return run_command("python3 -m venv venv", "Creating virtual environment")

def install_dependencies():
    """Install required dependencies"""
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    return run_command(f"{activate_cmd} && pip install -r requirements.txt", "Installing dependencies")

def create_config_file():
    """Create config file from template"""
    if os.path.exists("config_local.py"):
        print("✅ Configuration file already exists!")
        return True
    
    if os.path.exists("config_local.py.template"):
        try:
            with open("config_local.py.template", "r") as template:
                content = template.read()
            
            with open("config_local.py", "w") as config:
                config.write(content)
            
            print("✅ Configuration file created from template!")
            print("📝 Please edit config_local.py to add your API keys if needed.")
            return True
        except Exception as e:
            print(f"❌ Failed to create config file: {e}")
            return False
    else:
        print("⚠️  No config template found, skipping config file creation.")
        return True

def test_installation():
    """Test if the installation works"""
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    return run_command(f"{activate_cmd} && python run_app.py --mode basic", "Testing installation")

def main():
    print("=" * 60)
    print("Cox-Ingersoll-Ross Interest Rate Models - Setup")
    print("=" * 60)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Set up virtual environment
    if not setup_virtual_environment():
        print("❌ Setup failed at virtual environment creation!")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation!")
        sys.exit(1)
    
    # Create config file
    create_config_file()
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed at testing!")
        print("You can still try running the application manually.")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print()
    print("2. Run the application:")
    print("   python run_app.py --mode optimized")
    print()
    print("3. Or run examples:")
    print("   python run_app.py --examples")
    print()
    print("4. For more options:")
    print("   python run_app.py --help")
    print()
    print("Happy modeling! 📈")

if __name__ == "__main__":
    main()
