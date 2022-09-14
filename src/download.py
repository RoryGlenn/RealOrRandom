from requests import get
from os import path, mkdir, listdir
from sys import exit as sysexit
from logging import getLogger

logger = getLogger('root')

class Download:
    @staticmethod
    def get_data_filenames(data_path: str) -> list[str]:
        with open(data_path, "r") as file:
            filenames = file.readlines()
        return [f.replace("\n", "") for f in filenames]

    @staticmethod
    def download_data(
        url: str, files_to_download: list[str], download_path: str
    ) -> None:

        # create data directory if it doesn't exist
        if not path.exists(download_path):
            mkdir(download_path)

        for file in files_to_download:
            # if file is not in data folder, download it
            if file not in listdir(download_path):
                response = get(url + file, stream=True)
                if response.ok:
                    with open(download_path + "/" + file, "wb") as file:
                        for chunk in response.iter_content(chunk_size=256):
                            file.write(chunk)
                    logger.debug(f"File {file.name.split('/')[1]} downloaded succesfully")
                else:
                    logger.debug(
                        f"Error: status code: {response.status_code} can't download file {file}"
                    )
                    logger.debug(response.close())
                    sysexit(1)
