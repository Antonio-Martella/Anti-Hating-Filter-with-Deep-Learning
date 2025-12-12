#!/bin/bash
# setup_env.sh
# Setup Python environment and install dependencies based on OS

set -e

PYTHON_VERSION=3.11.8
VENV_NAME=venv

# Detect OS
OS_TYPE=$(uname)
if [ "$OS_TYPE" == "Darwin" ]; then
	    REQUIREMENTS_FILE="requirements_macos_arm.txt"
    else
	        REQUIREMENTS_FILE="requirements.txt"
fi

echo "Detected OS: $OS_TYPE"
echo "Using requirements file: $REQUIREMENTS_FILE"

# Function to install dependencies
install_deps() {
	    python -m pip install --upgrade pip
	        pip install -r $REQUIREMENTS_FILE
	}

	# Check if pyenv is available
	if command -v pyenv &> /dev/null; then
		    echo "pyenv found."
		        
		        if pyenv commands | grep -q "virtualenv"; then
				        echo "pyenv-virtualenv found."

					        # Install Python version if missing
						        if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
								            echo "Installing Python ${PYTHON_VERSION}..."
									                pyenv install ${PYTHON_VERSION}
											        fi

												        # Create virtualenv if missing
													        if ! pyenv virtualenvs --bare | grep -q "^${VENV_NAME}$"; then
															            echo "Creating virtualenv ${VENV_NAME}..."
																                pyenv virtualenv ${PYTHON_VERSION} ${VENV_NAME}
																		        fi

																			        # Activate virtualenv
																				        pyenv local ${VENV_NAME}
																					        install_deps
																						        echo "Setup complete with pyenv-virtualenv."
																							        echo "Activate with: pyenv activate ${VENV_NAME}"
																								        exit 0
																									    else
																										            echo "pyenv-virtualenv not found. Falling back to standard venv."
																											        fi
	fi

	# Fallback: standard venv
	echo "Using standard python -m venv."
	python3 -m venv ${VENV_NAME}

	# Activate the virtualenv in the current shell
	source ${VENV_NAME}/bin/activate
	install_deps
	echo "Setup complete with standard venv."
	echo "Virtual environment activated. To activate later: source ${VENV_NAME}/bin/activate"

