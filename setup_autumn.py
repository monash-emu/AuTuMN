from pathlib import Path
import os
import argparse

def is_windows():
    return os.name == 'nt'

def setup_autumn(update=False, server=False, force_jupyter=False, requirements='requirements.txt', only_forced=False):

    if only_forced:
        if force_jupyter:
            setup_jupyterlab_templates(force_jupyter)
        return
    
    if not update:
        if is_windows():
            patch_pywin32()
    
        os.system(f"pip install -r {requirements}")
        os.system('pip install -e ./')
        
        if not server:
            os.system('nbstripout --install')
            setup_jupyterlab_templates(force_jupyter)

    else:
        os.system(f'pip install --upgrade -r {requirements}')


def setup_jupyterlab_templates(force_templates=False):
    not_installed = os.system('jupyter labextension check jupyterlab_templates')

    if not_installed:
        os.system('jupyter labextension install jupyterlab_templates')
        os.system('jupyter serverextension enable --py jupyterlab_templates')

    jlab_config(force_templates)


def jlab_config(force_templates=False):
    home = Path.home()
    config_path = home / '.jupyter'
    config_path.mkdir(exist_ok=True)

    script_path = Path(__file__).resolve().parent
    nb_templates_path = script_path / 'notebooks' / 'templates'

    nbconf_file = config_path / 'jupyter_notebook_config.py'

    template_conf_str = None

    if nbconf_file.exists():
        print(f"Existing notebook config file found at {nbconf_file}")
        with open(nbconf_file, 'r') as conf_file_h:
            for line in conf_file_h.readlines():
                if "c.JupyterLabTemplates.template_dirs" in line:
                    template_conf_str = line

    config_buffer = [
        'c.JupyterLabTemplates.include_default = False\n',
        'c.JupyterLabTemplates.include_core_paths = False\n'
    ]

    write_templates = True

    if template_conf_str is not None:
        print(f"Templates already configured as {template_conf_str}")
        if force_templates:
            print("Force templates is True, overwriting")
        else:
            write_templates = False
    
    if write_templates:
        with open(nbconf_file, 'w') as conf_file_h:
            conf_file_h.writelines(config_buffer)
            template_conf_str = f"c.JupyterLabTemplates.template_dirs = [\'{nb_templates_path.as_posix()}\']\n"
            print(f"Writing template path as {template_conf_str}")
            conf_file_h.writelines([template_conf_str])


def patch_pywin32():
    print("Patching pywin32")
    os.system('conda install -y -c conda-forge pywin32')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--update", help="Update existing environment",
                        action="store_true")
    parser.add_argument("-r", "--requirements", default='requirements.txt', help="Requirements file to use")
    parser.add_argument("-s", "--server", help="Server install, no interactive components", action="store_true")
    parser.add_argument("--force-jupyter", action="store_true", help="Force overwrite existing Jupyter config")
    parser.add_argument("-OF", "--only-forced", action="store_true", help="Only perform forced actions")
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    setup_autumn(args.update, args.server, args.force_jupyter, args.requirements, args.only_forced)
