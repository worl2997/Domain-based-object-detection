import gdown
import zipfile

url = 'https://drive.google.com/uc?id=1GPueFgeVyHZBLc88fyjL6nSem_9lpAe9'
output = 'cityscape.tar.xz'
gdown.download(url, output, quiet=False)
dl_file = zipfile.ZipFile(output)
dl_file.extractall()