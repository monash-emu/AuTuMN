<div style="position: relative; padding-bottom: 66.66666666666666%; height: 0;"><iframe src="https://www.loom.com/embed/e7c00f7a1e454cb99cafc14c19dca330" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

# Step 1: Install Anaconda

## Anaconda setup for Windows

Open a PowerShell window as _Administrator_ by searching for it in your Windows menu. Run the following lines to install [Anaconda](https://www.anaconda.com/)

```powershell
# Install Chocolately package manager - https://chocolatey.org/install
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
# Use Chocolately to install Anaconda
choco install anaconda3
# Setup Anaconda to work with PowerShell
conda init powershell
```

## Anaconda setup for Linux

Download Anaconda and run the installer in a bash prompt:

```bash
# Download Anaconda
DOWNLOAD_URL=https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
wget -O /tmp/anaconda.sh $DOWNLOAD_URL
# Install Anaconda
cd /tmp/
bash anaconda.sh
# Setup Anaconda to work with bash
conda init bash
```

# Step 2: Set up Anaconda environment

In either bash (Linux) or PowerShell (Windows), run

```bash
# Create Autumn environment in Anaconda
conda create -n autumn python=3.6
# Activate autumn environment
conda activate autumn
```

Then, navigate to the autumn project directory and run:

```
pip install -r requirements.txt
```

# Step 3: Install PyCharm

Download and install Pycharm community edition [here](https://www.jetbrains.com/pycharm/download/#section=windows).

Once PyCharm is downloaded, create a new project with the Autumn folder as the target directory.
Then, select `File > Setttings > Project Interpreter` and select your Anaconda environment.

You should now be able to run the Autumn models.

# Troubleshooting

If you have an issue with a SQLite DLL not found on Windows, try the following steps:

- Visit the [SQLite website](https://www.sqlite.org/download.html)
- Download the 64-bit DLL (Precompiled Binaries for Windows)
- Place the DLL in your Anaconda environment's folder, in the "DLL" folder
