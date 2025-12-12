#!/bin/bash
# setup_env.sh
# Script to setup Python 3.11.8 environment with pyenv if available,
# otherwise falls back to standard venv.

set -e

PYTHON_VERSION=3.11.8
VENV_NAME=myproject-env

# Function to install dependencies in the current python environment
install_deps() {
	    python -m pip install --upgrade pip
	        pip install -r requirements.txt
	}

	# Check if pyenv is installed
	if command -v pyenv &> /dev/null; then
		    echo "pyenv found."
		        
		        # Check if pyenv-virtualenv is installed
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

																			            # Use local virtualenv
																				            pyenv local ${VENV_NAME}
																					            install_deps
																						            echo "Setup complete with pyenv-virtualenv."
																							            exit 0
																								        else
																										        echo "pyenv-virtualenv not found. Falling back to standard venv."
																											    fi
	fi

	# Fallback: standard venv
	echo "Using standard python -m venv."
	python3 -m venv ${VENV_NAME}
	source ${VENV_NAME}/bin/activate
	install_deps
	echo "Setup complete with standard venv."
