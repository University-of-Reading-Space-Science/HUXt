import os
import shutil
import subprocess
import sys
from pathlib import Path
from appdirs import user_data_dir

import huxt.huxt as h

def main():
    """Copy the master version of the examples notebook and then open in browser"""

    root_dir = Path(__file__).parent.parent
    nbk_master_path = root_dir.joinpath('notebooks', 'HUXt_examples.ipynb')

    nbk_out_dir = Path(user_data_dir(appname='huxt', appauthor=False), "notebooks")
    nbk_out_dir.mkdir(parents=True, exist_ok=True)
    nbk_out_path = nbk_out_dir.joinpath('HUXt_examples.ipynb')

    print(f"Copying {nbk_master_path} to {nbk_out_path}")
    shutil.copy(nbk_master_path, nbk_out_path)

    # Now open the copied notebook.
    print(f"Opening notebook: {nbk_out_path}")
    subprocess.run([sys.executable, "-m", "jupyter", "lab", nbk_out_path])

    return

if __name__ == '__main__':
    main()