# setup.py
from setuptools import setup, find_packages

setup(
    name="enhanced-voice-assistant",
    version="2.0.0",
    description="Enhanced voice assistant with personality, memory, and discrete mode",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "vosk>=0.3.45",
        "pvrecorder>=1.2.2",
        # Removed ollama dependency
    ],
    entry_points={
        "console_scripts": [
            "voice-assistant=main:main",
        ],
    },
    python_requires=">=3.8",
)