from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import os

from config import DATA_DIR


def download(drive_folder_id):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    files = drive.ListFile({'q': f"'{drive_folder_id}' in parents and trashed=false"}).GetList()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for i, file1 in enumerate(sorted(files, key=lambda x: x['title'])):
        print('Downloading {} from GDrive ({}/{})'.format(file1['title'], i, len(files)))
        file1.GetContentFile(os.path.join(DATA_DIR, file1['title']))
