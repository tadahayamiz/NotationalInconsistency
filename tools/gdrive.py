import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


TOOLS_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
GDRIVE_PATH2ID = {
    "": "0APBv_XbqYf3EUk9PVA",
    "data": "1TOR5JOKCuFWCEW_sPgOnVCSMH3pfx6_s",
    "transformer": "1sdTjVW6FjYspK-CEZOCEMurmvc1_h3BS",
    "graph/accuracy": "1H-lAkaxi1P-AAaFqxF73LVlZdepjEAZB",
    "graph/compare_surge": "18gggDsaWTil4HgXVWgxYC7LlEf7qMk_R",
    "graph/compare_thres": "1Nq_RKVMYEzhwfP4006DjClg9ym_aA5SU"
}

gauth = GoogleAuth(settings_file=TOOLS_DIR+"/gdrive_settings/settings.yaml")
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
def upgdrive(upped_file, title, mimetype, folder_path_or_id):
    if folder_path_or_id in GDRIVE_PATH2ID:
        folder_id = GDRIVE_PATH2ID[folder_path_or_id]
    else:
        folder_id = folder_path_or_id
    qstr = "title = \"\" and \"" + folder_id + "\" in parents and trashed=false"
    qstr = f'title = "{title}" and "{folder_id}" in parents and trashed=false'
    files = drive.ListFile({'q': qstr}).GetList()
    if len(files) > 0:
        file = files[0]
    else:
        file = drive.CreateFile({'title': title, 'mimeType': mimetype, 'parents': [{'id': folder_id}]})
    file.SetContentFile(upped_file)
    file.Upload()