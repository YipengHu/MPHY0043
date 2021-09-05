
import os
import shutil
import requests
import zipfile


DATA_PATH = 'data'

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)


print('Downloading...')
url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/jigsaws/video-meta.zip' 
r = requests.get(url,allow_redirects=True)
zip_file = os.path.join(DATA_PATH,'video-meta.zip')
_ = open(zip_file,'wb').write(r.content)


print('Extracting...')
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(DATA_PATH)
os.remove(zip_file)

print('JIGSAWS video data downloaded and extracted in: %s' % os.path.abspath(DATA_PATH))
