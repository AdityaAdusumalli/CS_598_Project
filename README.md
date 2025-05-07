# CS_598_Project

Overview : This code reproduces an experiment of the multi-label ECG classification expirements from the paper "In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis" by Nonaka and Seita (2021). This focuses on training and evaluating the ResNet-18 architecture on the "all" diagnostic statements task using a 1000-record subset of the PTB-XL dataset within a Python 3.9 environment managed by Poetry.


Setting Up Google Colab:
1) Open up a New Notebook in Google Colab
2) Set the Runtime->Change Runtime Type -> select t4 GPU

Limitations: 
1) During this reproduction attempt, I faced many issues when trying to process the full PTB-XL dataset (~21,000 records) as Google Colab T4 GPU environment has certain memory/duration limitations leading to out-of-memory errors. To achieve a proper result within the project timeframe and available resources, the scope was adjusted to utilize a 1000-record subset. Also, due to the limited subset placed by the limitations of the GPU, I manually inputed hyperparameters reported by the original authors to execute the core multi-label classification workflow (Experiment 1) for the baseline ResNet-18 model.


Commands/Steps to run in Colab:
You can follow the steps as seen in the cells posted in the ipynb file I uploaded.

1) Install Python and dependencies.
!sudo apt-get update -y
!sudo apt-get install python3.9 python3.9-dev python3.9-distutils libpython3.9-dev

2) Set Python 3.9 as default Python
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
!sudo update-alternatives --set python3 /usr/bin/python3.9

3) Download and install 'pip' specifically for Python 3.9
!curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
!python3.9 get-pip.py

4) Install poetry:
!python3.9 -m pip install poetry

5) Add Poetry's installation directory to the system PATH
import os
local_bin_path = os.path.expanduser('~/.local/bin')
if local_bin_path not in os.environ['PATH']:
    os.environ['PATH'] = f"{local_bin_path}:{os.environ['PATH']}"
print(f"Updated PATH: {os.environ['PATH']}")

6) Clone the 'dnn_ecg_comparison' repository. 
!git clone https://github.com/seitalab/dnn_ecg_comparison.git
%cd dnn_ecg_comparison

7) Modify the `pyproject.toml` file to update constraints for compatibility 
!sed -i 's/python = "^3.8"/python = "^3.9"/' pyproject.toml
!sed -i 's/scikit-learn = "0.23.2"/scikit-learn = "^1.0"/' pyproject.toml
!sed -i 's/Bottleneck = "1.3.2"/Bottleneck = "^1.3.4"/' pyproject.toml

8) Remove existing lock file and add project dependencies to use Poetry. 
!rm poetry.lock
!poetry add "ipython>=8.12.0,<9.0.0"
!poetry add matplotlib-inline

9) Install the project dependencies and navigate into the 'preparation' directory. 
!poetry install
%cd /content/dnn_ecg_comparison/
%cd preparation

10) Create directory structure for data and download the data.
!mkdir -p ../data/PTBXL/raw
!wget -O ../data/PTBXL/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip

11) Navigate into data directory, unzip the download, and rename the folder.
%cd /content/dnn_ecg_comparison/data/PTBXL/raw/
!unzip -o ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
!mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ptbxl

12) Move content to 'raw' directory, remove previous directory, enter the preperation directory
!mv ptbxl/* .
!rmdir ptbxl
%cd /content/dnn_ecg_comparison/preparation/

13)  Defines the corrected content for `preparation/utils.py`. Run the data preprocessing script for data split 1 on the 1000-record subset.

Second to last cell in ipynb^

15) Set up the `experiment/` directory structure. Write the correct content for `models.py`, `dataset.py`, and `resnet1d.py`. Create necessary `__init__.py` files for Python packaging. Writes the final, corrected `execute_clf_multilabel.py` script. Executes `!poetry run python execute_clf_multilabel.py resnet1d-18 cuda:0 1`. The output showing the training log and final Test AUC should appear below this cell.

- Last Cell in ipynb^


