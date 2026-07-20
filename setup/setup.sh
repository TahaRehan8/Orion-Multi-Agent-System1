@ -1,55 +0,0 @@
#!/bin/bash

cd "$(dirname "$0")"
cd ../..

if [ -f "setup/.env" ]; then
    export $(grep -v '^#' setup/.env | xargs)
fi

if [ ! -d ".git" ]; then
    if [ -z "$GIT_USERNAME" ] || [ -z "$GIT_PASSWORD" ]; then
        echo "Please set GIT_USERNAME and GIT_PASSWORD as env vars in setup/.env."
        exit 1
    fi
    git clone https://$GIT_USERNAME:$GIT_PASSWORD@github.com/Iris-Auto-ae/orion.git .
else
    git fetch origin
fi

git checkout dev
git pull origin dev

PYTHON_VERSION="3.12"
PY_FOUND=$(python3 --version 2>&1 | grep $PYTHON_VERSION)

if [ -z "$PY_FOUND" ]; then
    if command -v pyenv >/dev/null 2>&1; then
        pyenv install $PYTHON_VERSION -s
        pyenv local $PYTHON_VERSION
    else
        echo "Please install Python $PYTHON_VERSION (using pyenv or your system's package manager) and rerun this script."
        exit 1
    fi
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

if [ -f "setup/requirements.txt" ]; then
    pip install -r setup/requirements.txt
else
    echo "requirements.txt not found in setup/"
    exit 1
fi

if [ -f "orion.py" ]; then
    python orion.py
else
    echo "orion.py not found at root of orion directory."
    exit 1
fi