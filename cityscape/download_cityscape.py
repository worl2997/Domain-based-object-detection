import gdown

url = 'https://drive.google.com/uc?id=1GPueFgeVyHZBLc88fyjL6nSem_9lpAe9'
output = 'cityscape.tar.xz'
gdown.download(url, output, quiet=False)