# CS336 Spring 2024 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

0. Set up a virtual environment and install packages:

**Option A: Using conda (recommended):**
``` sh
conda create -n cs336_basics python=3.10 --yes
conda activate cs336_basics
pip install -e .'[test]'
```

**Option B: Using Python's built-in venv:**

*On macOS/Linux:*
``` sh
# Create virtual environment
python -m venv cs336_basics

# Activate virtual environment
source cs336_basics/bin/activate

# Install packages
pip install -e .'[test]'
```

*On Windows (Command Prompt):*
``` cmd
# Create virtual environment
python -m venv cs336_basics

# Activate virtual environment
cs336_basics\Scripts\activate

# Install packages
pip install -e .[test]
```

*On Windows (PowerShell):*
``` powershell
# Create virtual environment
python -m venv cs336_basics

# Activate virtual environment
cs336_basics\Scripts\Activate.ps1

# Install packages
pip install -e .[test]
```

1. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

2. Download the TinyStories data and a subsample of OpenWebText:

**On macOS/Linux:**
``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

**On Windows (PowerShell):**
``` powershell
# Create data directory
if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" }
cd data

# Download TinyStories data
curl -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Download and extract OpenWebText data
curl -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
Expand-Archive -Path owt_train.txt.gz -DestinationPath . -Force
curl -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz  
Expand-Archive -Path owt_valid.txt.gz -DestinationPath . -Force

cd ..
```

**On Windows (Command Prompt with curl):**
``` cmd
# Create data directory
if not exist "data" mkdir data
cd data

# Download TinyStories data
curl -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Download compressed OpenWebText data (you'll need to extract manually)
curl -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

# Note: Extract the .gz files manually using 7-Zip or similar tool
echo "Please extract owt_train.txt.gz and owt_valid.txt.gz manually using 7-Zip or similar"

cd ..
```

