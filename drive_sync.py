from __future__ import annotations
import io, os
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DB_NAME = "chats.db"

def _svc():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gdrive"]), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def _find_file_id(service, name):
    q = f"name = '{name}' and trashed = false"
    folder_id = st.secrets["gdrive"].get("folder_id")  # optional
    if folder_id:
        q += f" and '{folder_id}' in parents"
    r = service.files().list(q=q, fields="files(id,name)").execute()
    files = r.get("files", [])
    return files[0]["id"] if files else None

def download_db():
    service = _svc()
    fid = _find_file_id(service, DB_NAME)
    if not fid:
        return False
    req = service.files().get_media(fileId=fid)
    with io.FileIO(DB_NAME, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True

def upload_db():
    if not os.path.exists(DB_NAME):
        return False
    service = _svc()
    fid = _find_file_id(service, DB_NAME)
    media = MediaIoBaseUpload(open(DB_NAME, "rb"), mimetype="application/octet-stream", resumable=False)
    folder_id = st.secrets["gdrive"].get("folder_id")  # optional
    metadata = {"name": DB_NAME}
    if folder_id:
        metadata["parents"] = [folder_id]
    if fid:
        service.files().update(fileId=fid, media_body=media).execute()
    else:
        service.files().create(body=metadata, media_body=media, fields="id").execute()
    return True
