
import os
import shutil
import requests
import zipfile


DATA_PATH = 'data'

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)


print('Downloading and extracting data...')
url = 'https://github.com/yipenghu/datasets/archive/refs/heads/jigsaw.zip' 
r = requests.get(url,allow_redirects=True)
temp_file = 'temp.zip'
_ = open(temp_file,'wb').write(r.content)

with zipfile.ZipFile(temp_file, 'r') as zip_ref:
    zip_ref.extractall(DATA_PATH)
os.remove(temp_file)
print('Done.')

SOURCE_PATH = os.path.abspath(os.path.join(DATA_PATH,'datasets-jigsaw/video-meta'))
for item in os.listdir(SOURCE_PATH):
    shutil.move(os.path.join(SOURCE_PATH, item), os.path.join(DATA_PATH, item))
shutil.rmtree(os.path.abspath(os.path.join(DATA_PATH,'datasets-jigsaw')))

print('JIGSAWS video data downloaded and extracted: %s' % DATA_PATH)
