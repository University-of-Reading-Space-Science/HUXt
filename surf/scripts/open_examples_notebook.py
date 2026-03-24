import shutil
import subprocess
import sys
from pathlib import Path
from appdirs import user_data_dir


def main():
    """Copy the master version of the examples notebook and then open in browser"""

    root_dir = Path(__file__).parent.parent
    nbk_master_path = root_dir.joinpath('notebooks', 'SURF_examples.ipynb')

    nbk_out_dir = Path(user_data_dir(appname='surf', appauthor=False), "notebooks")
    nbk_out_dir.mkdir(parents=True, exist_ok=True)
    nbk_out_path = nbk_out_dir.joinpath('SURF_examples.ipynb')

    print(f"Copying {nbk_master_path} to {nbk_out_path}")
    shutil.copy(nbk_master_path, nbk_out_path)

    # Now open the copied notebook.
    print(f"Opening notebook: {nbk_out_path}")
    subprocess.run([sys.executable, "-m", "jupyter", "lab", nbk_out_path])

    return


if __name__ == '__main__':
    main()
