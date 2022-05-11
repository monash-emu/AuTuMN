# Step 1: Install Chocolatey package manager

Open a PowerShell window as _Administrator_ by searching for it in your Windows menu.
Note this will not work if you are using a terminal other than PowerShell.
Once installed, run the following line of code:

```powershell
Set-ExecutionPolicy Bypass -Scope CurrentUser -Force
$url = 'https://chocolatey.org/install.ps1'
iex ((New-Object System.Net.WebClient).DownloadString($url))
```
(further information about the Chocolatey installation manager is available at https://chocolatey.org/install)

# Step 2: Install Git, Git Large File Storage

In a PowerShell admin prompt, install [Git](https://git-scm.com) and [Git Large File Storage](https://git-lfs.github.com/) using Chocolatey.

```powershell
choco install git -y --force
choco install git-lfs -y --force
```

# Step 3: Install Anaconda

Approximate install time: 10 minutes.

## Anaconda setup for Windows

In a PowerShell admin prompt, install [Anaconda](https://www.anaconda.com/) using Chocolatey.

```powershell
# Use Chocolately to install Anaconda
choco install anaconda3 -y --force

# Update PATH environment variable so your computer knows where conda.exe is
$RegistryLoc = "Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment"
$CurrentPath = (Get-ItemProperty -Path $RegistryLoc -Name PATH).path
$NewPath = "$CurrentPath;C:\tools\Anaconda3\Scripts"
Set-ItemProperty -Path $RegistryLoc -Name PATH -Value $NewPath

# Close and re-open powershell as an administrator
# Setup Anaconda to work with PowerShell
# Run the following command using the Anaconda Powershell Prompt (can be found through the start menu)
conda init powershell

```

## Anaconda setup for Linux

Download Anaconda and run the installer in a bash prompt:

```bash
# Download Anaconda
DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget -O /tmp/conda.sh $DOWNLOAD_URL
# Install Anaconda
cd /tmp/
bash conda.sh
# Setup Anaconda to work with bash
# - accept license agreenment
# - default install location is okay
# - ask the installer to "initialize Miniconda" for you
. ~/.bashrc
conda config --set auto_activate_base false
```

# Step 4: Set up Anaconda environment

Close and re-open your shell. In either bash (Linux) or PowerShell (Windows), run

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

and finally

```
pip install -e ./
```

Now try run an Autumn model from the command line.

```powershell
# Ask a staff member for the autumn secret password
python -m autumn secrets read
# Build the input database
python -m autumn db build
# Run a model
python -m apps run covid malaysia --no-scenarios
```

# Step 5: Install PyCharm (Optional)

Approximate install time: 5 minutes.

You may want to use the same code editor as most of the EMU team - PyCharm.

Download and install Pycharm community edition [here](https://www.jetbrains.com/pycharm/download/#section=windows).

Once PyCharm is downloaded, create a new project with the Autumn folder as the target directory.
Then, select `File > Setttings > Project Interpreter` and select your Anaconda environment.

You should now be able to run the Autumn models via PyCharm.

# Step 6: Install remote libraries (Optional)

We have extra Python libraries that we use to run our code in the cloud (eg. high performance computing in AWS). You can get started with this as follows:

```powershell
pip install -r .\remote\requirements.txt
```

Then either set up your AWS profile with the AWS command line tool

```powershell
aws configure --profile autumn
# AWS Access Key ID [None]: <enter your actual key ID>
# AWS Secret Access Key [None]: <enter your actual secret access key>
# Default region name [None]: ap-southeast-2
# Default output format [None]: json
```

Or manually set your AWS environemnt variables

```powershell
$env:AWS_ACCESS_KEY_ID = "<enter your actual key ID>"
$env:AWS_SECRET_ACCESS_KEY = "<enter your actual secret access key>"
```

Now you should be able to use the `remote` module:

```powershell
python -m remote --help
python -m remote aws status
# Name         Type      Status    IP              Launched
# -----------  --------  --------  --------------  ------------
# buildkite-1  t3.small  running   54.153.241.26   2 months ago
# buildkite-2  t3.small  running   54.252.224.236  2 months ago
# buildkite-3  t3.small  running   13.236.184.15   2 months ago
# website      t2.nano   running   13.54.204.229   2 months ago
```

# Troubleshooting

If you have an issue with a SQLite DLL not found on Windows, try the following steps:

- Visit the [SQLite website](https://www.sqlite.org/download.html)
- Download the 64-bit DLL (Precompiled Binaries for Windows)
- Place the DLL in your Anaconda environment's folder, in the "DLL" folder

# Recreating Autumn conda environment

To re-create your autumn conda environment from scratch:

```powershell
conda activate base
conda env remove -n autumn -y
conda create -n autumn python=3.6 -y
conda activate autumn
pip install -r requirements.txt
```
