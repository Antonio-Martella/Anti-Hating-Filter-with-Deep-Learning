#!/bin/bash
# setup_env.sh
# Script to setup Python 3.11.8 environment with pyenv and install dependencies

# Exit on any error
set -e

# Desired Python version
PYTHON_VERSION=3.11.8
VENV_NAME=myproject-env

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null
then
	    echo "pyenv not found. Please install pyenv first."
	        exit 1
fi

# Install Python version if not already installed
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
	    echo "Installing Python ${PYTHON_VERSION}..."
	        pyenv install ${PYTHON_VERSION}
fi

# Check if pyenv-virtualenv is installed
if ! pyenv commands | grep -q "virtualenv"; then
	    echo "pyenv-virtualenv not found. Please install pyenv-virtualenv first."
	        exit 1
fi

# Create virtualenv if it doesn't exist
if ! pyenv virtualenvs --bare | grep -q "^${VENV_NAME}$"; then
	    echo "Creating virtualenv ${VENV_NAME}..."
	        pyenv virtualenv ${PYTHON_VERSION} ${VENV_NAME}
fi

# Set local pyenv version for this project
pyenv local ${VENV_NAME}

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete. Python version:"
python --version

