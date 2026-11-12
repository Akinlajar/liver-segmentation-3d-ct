
import os
import zipfile
import shutil
import pydicom
from tqdm import tqdm

def extract_zip_file(zip_file, extract_path):
    """Extracts the contents of a zip file to the specified target path."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def copy_file(source_file, target_file):
    """Copies a file from the source path to the target path."""
    shutil.copy2(source_file, target_file)


def rename_dicom_files(target_dir):
    """Renames DICOM files in the target directory using their InstanceNumber attribute."""
    for root, _, files in tqdm(os.walk(target_dir), desc="Renaming DICOM files"):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                dcm = pydicom.dcmread(file_path)
                ins_num = int(dcm.InstanceNumber)
                new_name = os.path.join(root, f"{ins_num:06d}.dcm")
                os.rename(file_path, new_name)
            except Exception:
                pass


def process_directory(source_dir, target_dir):
    """Processes the source directory: copies files, extracts zip files, and preserves the directory structure."""
    for root, dirs, files in tqdm(os.walk(source_dir), desc="Processing directories"):
        relative_path = os.path.relpath(root, source_dir)

        target_path = os.path.join(target_dir, relative_path)
        os.makedirs(target_path, exist_ok=True)

        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_path, file)

            if file.endswith('.zip'):
                extract_zip_file(source_file, target_path)
            else:
                copy_file(source_file, target_file)


if __name__ == "__main__":
    """Main entry point for the script. Defines source and target directories, and executes the workflow."""

    ### Define input and output paths
    src_dir = "data/raw/3Dircadb1"
    out_dir = "data/processed/3d_ircadb_images_masks_all_dicom"

    ### Reset output directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    ### Copy and extract files
    process_directory(src_dir, out_dir)

    ### Rename DICOM files in the target directory
    rename_dicom_files(out_dir)

    print(' Well Done! '.center(50, '='))
