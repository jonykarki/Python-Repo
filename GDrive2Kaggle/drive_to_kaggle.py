# created to help download files from drive to kaggle

from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import sys

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

def authorize_app():
    """
    Authorize the application using 'credentials.json'
    """
    creds = None
    # store the authorized token in a pickle
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # if no pickle, login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def get_all_folders(service):
    """
    Get all folders from the drive
    """
    page_token = None
    while True:
        response = service.files().list(q="mimeType = 'application/vnd.google-apps.folder'",
                                            spaces='drive',
                                            fields='nextPageToken, files(id, name)',
                                            pageToken=page_token).execute()
        for file in response.get('files', []):
            # Process change
            print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

def get_file_ids_from_folder(service, folder_id):
    file_ids = []
    response = service.files().list(q=f"mimeType != 'application/vnd.google-apps.folder'",
                                    spaces='drive',
                                    fields='files(id,name,parents)').execute()
    for file in response.get('files', []):        
        if file['parents'][0] == folder_id:
            file_ids.append(file.get('id'))
    return file_ids

def download_links_kaggle(service, folder_id):
    file_ids = get_file_ids_from_folder(service, folder_id)
    for id in file_ids:
        print(f'!gdown https://drive.google.com/uc?id={id}')

def main(folder_id):
    creds = authorize_app()

    service = build('drive', 'v3', credentials=creds)
    # make sure the folder is shared
    download_links_kaggle(service, folder_id)

if __name__ == '__main__':
    url = sys.argv[1]
    folder_id = str(url).split('/')[-1]
    main(folder_id)