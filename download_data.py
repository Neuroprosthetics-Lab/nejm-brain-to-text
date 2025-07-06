"""
Run this file to download data from Dryad and unzip the zip files. Downloaded files end
up in this repostitory's data/ directory.

First create the b2txt25 conda environment. Then in a Terminal, at this repository's
top-level directory (nejm-brain-to-text/), run:

conda activate b2txt25
python download_data.py
"""

import sys
import os
import urllib.request
import json
import zipfile


########################################################################################
#
# Helpers.
#
########################################################################################


def display_progress_bar(block_num, block_size, total_size, message=""):
    """"""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    """"""
    DRYAD_DOI = "10.5061/dryad.dncjsxm85"

    ## Make sure the command is being run from the right place and we can see the data/
    ## directory.

    DATA_DIR = "data/"
    data_dirpath = os.path.abspath(DATA_DIR)
    assert os.getcwd().endswith(
        "nejm-brain-to-text"
    ), f"Please run the download command from the nejm-brain-to-text directory (instead of {os.getcwd()})"
    assert os.path.exists(
        data_dirpath
    ), "Cannot find the data directory to download into."

    ## Get the list of files from the latest version on Dryad.

    DRYAD_ROOT = "https://datadryad.org"
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"][
        "stash:files"
    ]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]

    ## Download each file into the data directory (and unzip for certain files).

    for file_info in file_infos:
        filename = file_info["path"]

        if filename == "README.md":
            continue

        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"

        download_to_filepath = os.path.join(data_dirpath, filename)

        urllib.request.urlretrieve(
            download_url,
            download_to_filepath,
            reporthook=lambda *args: display_progress_bar(
                *args, message=f"Downloading {filename}"
            ),
        )
        sys.stdout.write("\n")

        # If this file is a zip file, unzip it into the data directory.

        if file_info["mimeType"] == "application/zip":
            print(f"Extracting files from {filename} ...")
            with zipfile.ZipFile(download_to_filepath, "r") as zf:
                zf.extractall(data_dirpath)

    print(f"Download complete. See data files in {data_dirpath}\n")


if __name__ == "__main__":
    main()
