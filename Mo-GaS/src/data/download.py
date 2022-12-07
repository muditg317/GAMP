from src.utils.config import *
import os
from subprocess import call

url_ph = 'https://zenodo.org/record/3451402/files/'
urls = [url_ph+'action_enums.txt']
urls += [url_ph+game+'.zip' for game in GAMES_FOR_TRAINING]

# download data using wget
for url in urls:
    file_name = url.split('/')[-1]
    file_out_name = os.path.join(RAW_DATA_DIR, file_name)
    if not os.path.exists(file_out_name):
        args_string = 'wget {} -O {}'.format(url, file_out_name)
        args = args_string.split(' ')
        call(args)
        # unzips files at RAW_DATA_DIR
        if file_out_name.__contains__('.zip'):
            unzip_str = 'unzip -n {} -d {}'.format(file_out_name, RAW_DATA_DIR)
            unzip_args = unzip_str.split(' ')
            call(unzip_args)


print(f"Data created at {PROC_DATA_DIR} and {INTERIM_DATA_DIR}")
print(f"Data should now be preprocessed using\n\tpython src/data/process.py")
