# Installation Guide
## Amazon Best Seller Rank Prediction using Multimodal Analysis

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Project Download](#project-download)
4. [Environment Setup](#environment-setup)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)
8. [Optional Components](#optional-components)

---

## System Requirements

### Minimum Requirements
- **Operating System**: macOS 10.15+, Windows 10+, or Linux (Ubuntu 20.04+)
- **RAM**: 8 GB (16 GB recommended for image processing)
- **Storage**: 10 GB free space
- **Internet**: Required for package downloads and API access

### Recommended Specifications
- **RAM**: 16 GB or more
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional, but beneficial for image processing (CUDA-compatible)
- **Storage**: SSD with 20 GB free space

### Software Dependencies
- Python 3.10 or higher
- Anaconda or Miniconda
- Git
- Jupyter Notebook/Lab

---

## Prerequisites Installation

### Step 1: Install Anaconda/Miniconda

#### macOS

1. **Download Miniconda**:
   - Visit: https://docs.conda.io/en/latest/miniconda.html
   - Download the macOS installer (Python 3.10+)

2. **Install via Terminal**:
   ```bash
   cd ~/Downloads
   bash Miniconda3-latest-MacOSX-x86_64.sh
   ```

3. **Follow the installation prompts**:
   - Press ENTER to review the license
   - Type `yes` to accept
   - Press ENTER to confirm the installation location
   - Type `yes` to initialize conda

4. **Restart Terminal** or run:
   ```bash
   source ~/.zshrc
   ```

5. **Verify Installation**:
   ```bash
   conda --version
   ```
   Expected output: `conda 23.x.x` or similar

#### Windows

1. **Download Anaconda**:
   - Visit: https://www.anaconda.com/download
   - Download the Windows installer (Python 3.10+)

2. **Run the Installer**:
   - Double-click the downloaded `.exe` file
   - Click "Next" through the setup wizard

3. **Installation Options**:
   - ✅ Install for "Just Me" (recommended)
   - ✅ Add Anaconda to PATH environment variable
   - ✅ Register Anaconda as default Python

4. **Complete Installation**:
   - Click "Install"
   - Wait for completion (~5-10 minutes)
   - Click "Finish"

5. **Verify Installation**:
   - Open "Anaconda Prompt" from Start Menu
   ```cmd
   conda --version
   ```
   Expected output: `conda 23.x.x` or similar

### Step 2: Install Git

#### macOS

**Option 1: Using Homebrew (Recommended)**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
```

**Option 2: Direct Download**
1. Visit: https://git-scm.com/download/mac
2. Download and install the package

**Verify Installation**:
```bash
git --version
```

#### Windows

1. **Download Git**:
   - Visit: https://git-scm.com/download/windows
   - Download the installer

2. **Run the Installer**:
   - Accept default options
   - Select "Git Bash Here" option
   - Use default text editor

3. **Verify Installation**:
   - Open Command Prompt or Git Bash
   ```cmd
   git --version
   ```

### Step 3: Install Visual Studio Code (Optional but Recommended)

1. **Download VS Code**:
   - Visit: https://code.visualstudio.com/
   - Download for your operating system

2. **Install VS Code**:
   - macOS: Drag to Applications folder
   - Windows: Run the installer

3. **Install Extensions**:
   - Open VS Code
   - Click Extensions icon (left sidebar)
   - Install:
     - **Python** (Microsoft)
     - **Jupyter** (Microsoft)
     - **Pylance** (Microsoft)

---

## Project Download

### Method 1: Clone from GitHub

1. **Open Terminal/Command Prompt**

2. **Navigate to desired directory**:
   ```bash
   cd ~/Desktop  # macOS/Linux
   cd C:\Users\YourName\Desktop  # Windows
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/DylanSuniaga/Forecasting-Amazon-Sales-with-Multimodal-Data.git
   ```

4. **Navigate into project directory**:
   ```bash
   cd Forecasting-Amazon-Sales-with-Multimodal-Data
   ```

### Method 2: Download ZIP

1. Visit the GitHub repository
2. Click green "Code" button
3. Select "Download ZIP"
4. Extract to desired location
5. Open Terminal/Command Prompt in extracted folder

---

## Environment Setup

### Step 1: Create Conda Environment

#### macOS/Linux

1. **Navigate to project directory**:
   ```bash
   cd /path/to/cis_amazon_forecast_proj
   ```

2. **Create environment from YAML file**:
   ```bash
   conda env create -f environment.yml
   ```
   
   This process may take 5-15 minutes depending on your internet speed.

3. **Activate the environment**:
   ```bash
   conda activate amazon-forecast
   ```

4. **Verify activation**:
   - Your terminal prompt should now show `(amazon-forecast)` at the beginning

#### Windows

1. **Open Anaconda Prompt**

2. **Navigate to project directory**:
   ```cmd
   cd C:\path\to\cis_amazon_forecast_proj
   ```

3. **Create environment from YAML file**:
   ```cmd
   conda env create -f environment.yml
   ```

4. **Activate the environment**:
   ```cmd
   conda activate amazon-forecast
   ```

### Step 2: Verify Package Installation

Run the following command to check installed packages:

```bash
conda list
```

**Key packages to verify**:
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- opencv-python >= 4.7.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- jupyter >= 1.0.0
- textstat >= 0.7.0

### Step 3: Install Additional Dependencies (if needed)

If any packages are missing:

```bash
conda activate amazon-forecast
pip install opencv-python textstat requests python-dotenv
```

---

## Configuration

### Step 1: Create Configuration Files

#### Create .env File for API Credentials

1. **Create a new file named `.env` in the project root**:
   ```bash
   touch .env  # macOS/Linux
   type nul > .env  # Windows
   ```

2. **Edit the .env file** (use any text editor):
   ```
   CLIENT_ID=your_amazon_sp_api_client_id
   CLIENT_SECRET=your_amazon_sp_api_client_secret
   REFRESH_TOKEN=your_amazon_sp_api_refresh_token
   ```

3. **Save the file**

> **Note**: If you don't have Amazon SP-API credentials, you can skip this step. 
> The pre-downloaded data in the `data/` directory allows you to run most analyses 
> without API access.

### Step 2: Verify Data Files

Check that the following files exist in the `data/` directory:

```bash
ls data/
```

Expected files:
- ✅ `17k_products_amazon_data.csv`
- ✅ `all_keywords_merged.csv`
- ✅ `data_with_scraper.csv`
- ✅ `batches/` directory with category files
- ✅ `image_analysis_data/` directory
- ✅ `images_amz/` directory (or `images_amz.zip`)

#### Extract Images (if needed)

If `images_amz/` doesn't exist but `images_amz.zip` does:

```bash
unzip data/images_amz.zip -d data/
```

### Step 3: Configure Jupyter

1. **Activate environment**:
   ```bash
   conda activate amazon-forecast
   ```

2. **Add environment to Jupyter**:
   ```bash
   python -m ipykernel install --user --name=amazon-forecast
   ```

3. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Or start Jupyter Lab** (recommended):
   ```bash
   jupyter lab
   ```

---

## Verification

### Step 1: Test Python Environment

1. **Create a test script**:
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost, cv2; print('All packages imported successfully!')"
   ```

   Expected output: `All packages imported successfully!`

### Step 2: Test Jupyter Notebook

1. **Start Jupyter**:
   ```bash
   conda activate amazon-forecast
   jupyter notebook
   ```

2. **Create a new notebook**:
   - Click "New" → "amazon-forecast"

3. **Run test code**:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Load sample data
   df = pd.read_csv('data/17k_products_amazon_data.csv')
   print(f"Dataset loaded: {len(df)} rows")
   print(df.head())
   ```

4. **Expected output**:
   ```
   Dataset loaded: 17375 rows
   [DataFrame preview]
   ```

### Step 3: Run Quick Analysis

1. **Navigate to notebooks**:
   ```bash
   cd notebooks/01-Base Model with Visuals/
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook bsr_analysis.ipynb
   ```

3. **Run all cells** (Kernel → Restart & Run All)

4. **Verify outputs**:
   - Model training completes successfully
   - Plots display correctly
   - No error messages

---

## Troubleshooting

### Issue 1: Conda Not Recognized

**Symptom**: `conda: command not found`

**Solution (macOS/Linux)**:
```bash
export PATH="$HOME/miniconda3/bin:$PATH"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Solution (Windows)**:
- Open "Environment Variables" in System Settings
- Add Anaconda path to PATH variable
- Restart Command Prompt

### Issue 2: Environment Creation Fails

**Symptom**: Package conflicts during `conda env create`

**Solution 1: Update Conda**:
```bash
conda update -n base -c defaults conda
```

**Solution 2: Create environment manually**:
```bash
conda create -n amazon-forecast python=3.10
conda activate amazon-forecast
pip install -r requirements.txt  # if available
```

### Issue 3: Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'cv2'`

**Solution**:
```bash
conda activate amazon-forecast
pip install opencv-python
```

### Issue 4: Jupyter Kernel Not Found

**Symptom**: Environment doesn't appear in Jupyter kernel list

**Solution**:
```bash
conda activate amazon-forecast
python -m ipykernel install --user --name=amazon-forecast --display-name "Python (amazon-forecast)"
```

### Issue 5: Memory Errors

**Symptom**: Kernel dies when loading large datasets

**Solutions**:
1. **Increase swap space** (OS-level)
2. **Use Parquet files instead of CSV**:
   ```python
   df = pd.read_parquet('data/all_keywords_merged.parquet')
   ```
3. **Process data in chunks**:
   ```python
   for chunk in pd.read_csv('file.csv', chunksize=1000):
       process(chunk)
   ```

### Issue 6: OpenCV Import Error (macOS)

**Symptom**: `ImportError: libpng16.16.dylib not found`

**Solution**:
```bash
brew install libpng
pip uninstall opencv-python
pip install opencv-python
```

### Issue 7: Permission Denied Errors

**Symptom**: Cannot write to directories

**Solution (macOS/Linux)**:
```bash
sudo chmod -R 755 /path/to/project
```

**Solution (Windows)**:
- Right-click project folder → Properties → Security
- Grant full control to your user account

---

## Optional Components

### GPU Support for Image Processing (Optional)

If you have an NVIDIA GPU and want to accelerate image processing:

1. **Install CUDA Toolkit**:
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Download and install for your OS

2. **Install PyTorch with CUDA**:
   ```bash
   conda activate amazon-forecast
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. **Verify GPU access**:
   ```python
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   print(f"GPU device: {torch.cuda.get_device_name(0)}")
   ```

### VS Code Integration (Optional)

1. **Open project in VS Code**:
   ```bash
   code /path/to/cis_amazon_forecast_proj
   ```

2. **Select Python interpreter**:
   - Press `Cmd/Ctrl + Shift + P`
   - Type "Python: Select Interpreter"
   - Choose "amazon-forecast" environment

3. **Open Jupyter notebooks**:
   - Click any `.ipynb` file
   - Select "amazon-forecast" kernel in top-right

---

## Next Steps

After successful installation:

1. ✅ Read `USER_MANUAL.md` for usage instructions
2. ✅ Review `README.txt` for directory structure
3. ✅ Explore notebooks in order (00 → 01 → 02 → 03 → 04)
4. ✅ Check `reports/dataset_audit.md` for data validation

---

## Support

If you encounter issues not covered in this guide:

**Email**: dsuniaga001@gmail.com

**Common Resources**:
- Conda Documentation: https://docs.conda.io/
- Jupyter Documentation: https://jupyter.org/documentation
- Python Documentation: https://docs.python.org/3/

---

**Installation Guide Version**: 1.0  
**Last Updated**: December 2025  
**Author**: Dylan Suniaga

