#!/usr/bin/env python
"""Setup script for claude-code-api package."""

import os
from setuptools import setup, find_packages

setup(
    name="claude-code-api",
    version="1.0.0",
    description="OpenAI-compatible API gateway for Claude Code with streaming support",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Claude Code API Team",
    url="https://github.com/claude-code-api/claude-code-api",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "pydantic>=2.9.0",
        "httpx>=0.27.0",
        "aiofiles>=24.1.0",
        "structlog>=24.4.0",
        "python-multipart>=0.0.12",
        "pydantic-settings>=2.6.0",
        "sqlalchemy>=2.0.35",
        "aiosqlite>=0.20.0",
        "alembic>=1.13.3",
        "passlib[bcrypt]>=1.7.4",
        "python-jose[cryptography]>=3.3.0",
        "python-dotenv>=1.0.1",
        "openai>=1.54.0",
    ],
    extras_require={
        "test": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=6.0.0",
            "httpx>=0.27.0",
            "pytest-mock>=3.14.0",
        ],
        "dev": [
            "black>=24.10.0",
            "isort>=5.13.0",
            "flake8>=7.1.0",
            "mypy>=1.13.0",
            "pre-commit>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "claude-code-api=claude_code_api.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
)
