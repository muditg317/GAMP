function force_source {
  # echo "force source -- $#: $@|"
  sourced=0
  if [ -n "$ZSH_EVAL_CONTEXT" ]; then
    # echo "zsh"
    case $ZSH_EVAL_CONTEXT in *:file) sourced=1;; esac
  elif [ -n "$BASH_VERSION" ]; then
    # echo "bash"
    if ! [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
      # echo "Hey, you should source this script, not execute it!"
      # exit 1
      sourced=1
    fi
    # (return 0 2>/dev/null) && sourced=1
  else # All other shells: examine $0 for known shell binary filenames
    # Detects `sh` and `dash`; add additional shell filenames as needed.
    # echo "other shell"
    case ${0##*/} in dash|-dash|bash|-bash|ksh|-ksh|sh|-sh) sourced=1;; esac
  fi
  # echo "checked: $sourced"
  if [ "$sourced" != "1" ]; then
    echo -e "You must use source for "\`$(basename $1)\`"\n\t. $(basename $1)"
    exit # not return because "you can only return from a funtion or sourced script"
  fi
}
force_source $0

function handler {
  echo -e "Interrupted, exiting..."
  return
}
trap handler INT

echo "Exiting/interrupting this script (ctrl-c) may behave unexpectedly since it's sourced. Sorry!"


# ensure script run from root of repository
if [ ! -f "setup.py" ]; then
  # echo ${BASH_SOURCE[0]}
  echo "You must run this script from the root of the Mo-GaS repository: $(dirname ${BASH_SOURCE[0]})"
  use_cd=0
  while true; do
      read -p "Do you want to cd to the root of the Mo-GaS repository ([y]/n)? " yn
      case $yn in
          "" | [Yy]* ) use_cd=1; break;;
          [Nn]* ) break;;
          * ) echo "Please answer yes or no.";;
      esac
  done
  if [ $use_cd -eq 0 ]; then
    return
  fi
  echo "Changing directory to $(dirname ${BASH_SOURCE[0]})"
  cd $(dirname ${BASH_SOURCE[0]})
fi

# ensure conda is installed
if ! command -v conda &> /dev/null; then
  echo "conda not found, please install it first"
  return
fi

# create conda environment
echo -e "\nConfiguring conda environment...\n"

env_name="mogas"

if conda env list | grep -q $env_name; then
  while conda env list | grep -q $env_name; do
    echo -e "conda environment "\`$env_name\`" already exists..."

    use_existing=0
    while true; do
        read -p "Do you want to install deps into the $env_name environment ([y]/n)? " yn
        case $yn in
            "" | [Yy]* ) use_existing=1; break;;
            [Nn]* ) break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    if [ $use_existing -eq 1 ]; then
      break
    fi
  

    while true; do
      echo
      read -p "What should the conda environment be called instead ([Mo-GaS])? " env_name
      case $env_name in
        "") env_name="Mo-GaS"; break;;
      esac
    done
  done
fi

if ! conda env list | grep -q $env_name; then
  echo "Creating conda environment "\`$env_name\`""
  conda create -n $env_name
  if conda env list | grep -q $env_name; then
    echo "conda environment "\`$env_name\`" created"
  else
    echo -e "Failed to create conda environment "\`$env_name\`".\n\tExiting..."
    return
  fi
fi

# activate conda environment
echo -e "\nActivating conda environment "\`$env_name\`""
conda activate $env_name

# install dependencies
echo -e "\nInstalling dependencies\n"

echo "Installing PyTorch"
use_cuda=0
while true; do
    read -p "Do you have a cuda-enabled GPU ([y]/n)? " yn
    case $yn in
        "" | [Yy]* ) use_cuda=1; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

if [ $use_cuda -eq 1 ]; then
  echo "Installing PyTorch with cuda"

  echo -e "\t++  conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia"
  conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

  echo "Verifying cuda installation"
  if ! [ "$(python -c "import torch; print(torch.cuda.is_available())")" = "True" ]; then
    echo -e "\tFailed to install cuda, exiting..."
    return
  else
    echo -e "\tPyTorch w/ cuda installed successfully"
  fi
else
  echo "Installing PyTorch without cuda (cpu only)"

  echo -e "\t++  conda install pytorch torchvision cpuonly -c pytorch"
  conda install pytorch torchvision cpuonly -c pytorch
fi

echo -e "\nInstalling other python dependencies"
echo -e "\t++  conda install click Sphinx coverage flake8 \"python-dotenv>=0.5.1\" tqdm PyYAML numpy pandas seaborn matplotlib scipy scikit-learn h5py tensorboard"
conda install click Sphinx coverage flake8 "python-dotenv>=0.5.1" tqdm PyYAML numpy pandas seaborn matplotlib scipy scikit-learn h5py tensorboard

echo -e "\nInstalling OpenCV via pip (using conda causes dependency conflicts)"
if ! command -v pip | grep -q "envs/$env_name"; then
  echo -e "\tpip not running from conda environment, exiting..."
  return
fi
echo -e "\t++  pip install opencv-python"
pip install opencv-python

echo -e "\nInstalling local directory as module via pip"
echo -e "\t++  pip install -e ."
pip install -e .

echo -e "\nInstalling jupyter notebook"
echo -e "\nInstalling jupyter tools / extensions"
use_jupyter=0
while true; do
    read -p "Do you plan to use jupyter notebooks ([y]/n)? " yn
    case $yn in
        "" | [Yy]* ) use_jupyter=1; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

if [ $use_jupyter -eq 1 ]; then
  use_vscode_jupyter=0
  while true; do
      read -p "Do you plan to use the ipynb support within VSCode ([y]/n)? " yn
      case $yn in
          "" | [Yy]* ) use_vscode_jupyter=1; break;;
          [Nn]* ) break;;
          * ) echo "Please answer yes or no.";;
      esac
  done

  if [ $use_vscode_jupyter -eq 0 ]; then
    echo -e "\nInstalling jupyter into conda environment"
    echo -e "\t++  conda install -c anaconda jupyter"
    conda install -c anaconda jupyter
  fi

  echo -e "\nInstalling jupyter extensions"
  echo -e "\t++  conda install -n $env_name ipykernel --update-deps --force-reinstall"
  conda install -n $env_name ipykernel --update-deps --force-reinstall
  echo -e "\t++  conda install -c conda-forge ipympl"
  conda install -c conda-forge ipympl
fi

echo -e "\n\nDone setting up Mo-GaS environment!\n"


echo -e "To get started, follow the instructions in the README.md file\n"

# remove trap
trap - INT