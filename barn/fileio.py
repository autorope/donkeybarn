import requests
from tqdm import tqdm
import tarfile
import os


def download_file(url, filepath, reload=False):
    if not os.path.exists(filepath) or reload == True:
        with open(filepath, "wb") as handle:
            response = requests.get(url, stream=True)
            for data in tqdm(response.iter_content()):
                handle.write(data)
    else:
        print('File already exists.')
    return filepath


def extract_file(in_filepath, out_dir):
    if in_filepath.endswith("tar.gz"):
        tar = tarfile.open(in_filepath, "r:gz")
        tar.extractall(path=out_dir)
        tar.close()
    elif in_filepath.endswith("tar"):
        tar = tarfile.open(in_filepath, "r:")
        tar.extractall(path=out_dir)
        tar.close()

    out_filepath = os.path.join(out_dir, os.path.basename(in_filepath))
    assert (os.path.exists(out_filepath))
    return out_filepath