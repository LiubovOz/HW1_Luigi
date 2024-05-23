# Импортируем все необходимые библиотеки
import gzip
import io
import luigi
import os
import pandas as pd
import requests
import shutil
import tarfile

from contextlib import closing

# Определяем константы для URL-адреса удаленного сервера,
# имени набора данных, размера фрагмента и пути к файлу набора данных.
DEFAULT_REMOTE = "https://www.ncbi.nlm.nih.gov/geo/download/?format=file"
DEFAULT_DATASET_NAME = "GSE68849"
CHUNK_SIZE = 8192

DEFAULT_DATASET_LOCATION = f"raw_dataset_{DEFAULT_DATASET_NAME}.tar"

# Создаем функцию для создания URL-адреса
def make_dataset_url(dataset_name, remote=DEFAULT_REMOTE):
    return remote + "&acc=" + dataset_name

# Создаем функцию  загрузки и сохранения датасета
def download_dataset(dataset_name, output,
                     remote=DEFAULT_REMOTE):
    dataset_url = make_dataset_url(dataset_name, remote)

    with requests.get(dataset_url, stream=True) as request:
        request.raise_for_status()

        for chunk in request.iter_content(chunk_size=CHUNK_SIZE):
            output.write(chunk)

# Создадим класс, который является задачей Luigi и отвечает за загрузку датасета
class FetchDatasetTask(luigi.Task):
    dataset_name = luigi.Parameter(default=DEFAULT_DATASET_NAME)
    dataset_target = luigi.Parameter(default=DEFAULT_DATASET_LOCATION)
    remote = luigi.Parameter(default=DEFAULT_REMOTE)

    def output(self):
        return luigi.LocalTarget(self.dataset_target)

    def run(self):
        with open(self.dataset_target, "wb") as output:
            download_dataset(self.dataset_name, output, self.remote)

# Создадим функцию для извлеченя содержимого TAR-файла в указанную директорию
def extract_tar(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with tarfile.open(input_path) as archive:
        archive.extractall(path=output_dir)

# Создадим функцию для извлечения соржимого архива
def extract_gzip(input_path, output):
    with closing(gzip.open(input_path, "rb")) as compressed_file:
        shutil.copyfileobj(compressed_file, output)


GZIP_EXTENSION = ".gz"

# Создадим функцию, которая извлекает содержимое TAR-файла и извлекает все файлы .gz внутри него.
def extract_dataset(input_path, output_dir):
    extract_tar(input_path, output_dir)

    directory_items = os.listdir(output_dir)

    for compressed_filename in directory_items:
        if not compressed_filename.endswith(GZIP_EXTENSION):
            continue

        compressed_path = os.path.join(output_dir, compressed_filename)
        extracted_path = compressed_path.removesuffix(GZIP_EXTENSION)
        with open(extracted_path, "wb") as output:
            extract_gzip(compressed_path, output)
        os.remove(compressed_path)

# Создадим класс, который является задачей Luigi и отвечает за извлечение датасета
class ExtractDatasetArchiveTask(luigi.Task):
    dataset_name = luigi.Parameter(default=DEFAULT_DATASET_NAME)
    dataset_target = luigi.Parameter(default=DEFAULT_DATASET_LOCATION)
    remote = luigi.Parameter(default=DEFAULT_REMOTE)
    dataset_directory = luigi.Parameter(
        default="extracted_" + DEFAULT_DATASET_NAME
    )

    def output(self):
        return luigi.LocalTarget(self.dataset_directory)

    def requires(self):
        return FetchDatasetTask(
            dataset_name=self.dataset_name,
            dataset_target=self.dataset_target,
            remote=self.remote,
        )

    def run(self):
        archive_path = self.input().path
        try:
            os.makedirs(self.dataset_directory, exist_ok=True)
            extract_dataset(archive_path, self.dataset_directory)
        finally:
            os.remove(archive_path)

# Создаем функцию, которая читает файл набора данных и возвращает словарь DataFrame
def parse_dataset_file(ds_file):
    dfs = {}

    write_key = None
    fio = io.StringIO()

    for line in ds_file.readlines():
        if line.startswith("["):
            if write_key:
                fio.seek(0)
                header = None if write_key == "Heading" else "infer"
                dfs[write_key] = pd.read_csv(fio, sep="\t", header=header)
            fio = io.StringIO()
            write_key = line.strip("[]\n")
            continue
        if write_key:
            fio.write(line)
    fio.seek(0)
    dfs[write_key] = pd.read_csv(fio, sep="\t")

    return dfs

# Создаем константу с именами столбцов, которые нужно удалить из датасета
COLUMNS_TO_REMOVE = [
    "Definition", "Ontology_Component", "Ontology_Process",
    "Ontology_Function", "Synonyms", "Obsolete_Probe_Id",
    "Probe_Sequence",
]
# Создаем константу с именем таблицы, из которой нужно удалить столбцы.
TABLE_TO_REMOVE_COLUMNS = "Probes"

# Создаем функцию, которая удаляет указанные столбцы из таблицы
def truncate_dataset_columns(dataset,
                             table_to_remove_cols=TABLE_TO_REMOVE_COLUMNS,
                             columns_to_remove=COLUMNS_TO_REMOVE):
    dataset[table_to_remove_cols] = dataset[table_to_remove_cols].drop(
        columns=columns_to_remove,
    )
    return dataset
# Создаем класс, который отвечает за удаление ненужных столбцов из датасета
class TruncateDatasetTask(luigi.Task):
    dataset_name = luigi.Parameter(default=DEFAULT_DATASET_NAME)
    dataset_target = luigi.Parameter(default=DEFAULT_DATASET_LOCATION)
    remote = luigi.Parameter(default=DEFAULT_REMOTE)
    dataset_directory = luigi.Parameter(
        default="extracted_" + DEFAULT_DATASET_NAME
    )
    table_to_remove_cols = luigi.Parameter(default=TABLE_TO_REMOVE_COLUMNS)
    columns_to_remove = luigi.Parameter(default=COLUMNS_TO_REMOVE)

    def output(self):
        return luigi.LocalTarget(self.dataset_target)

    def requires(self):
        return ExtractDatasetArchiveTask(
            dataset_name=self.dataset_name,
            dataset_target=self.dataset_target,
            remote=self.remote,
            dataset_directory=self.dataset_directory,
        )

    def run(self):
        for dataset_file in os.listdir(self.input().path):
            dataset_file_path = os.path.join(self.input().path, dataset_file)

            truncation_result = {}
            with open(dataset_file_path, "r") as ds_input:
                parse_result = parse_dataset_file(ds_input)
                truncation_result = truncate_dataset_columns(parse_result)

            part_directory = dataset_file_path.removesuffix(".txt")
            os.makedirs(part_directory, exist_ok=True)

            for table_name, table in truncation_result.items():
                table_path = os.path.join(part_directory, table_name + ".tsv")
                table.to_csv(table_path, sep="\t")

            os.remove(dataset_file_path)

# Запускаем Luigi, чтобы выполнить все задачи
if __name__ == "__main__":
    luigi.build(
        [TruncateDatasetTask(dataset_name="GSE68849")],
        local_scheduler=True,
    )
