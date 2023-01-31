import os
import time
import re


def bs_description(batch_size):
    bs = f"    bs {batch_size}:"
    return f"{bs:<15}"


def get_trailing_number(string):
    nb = re.search(r'\d+$', string)
    return int(nb.group()) if nb else None


class BenchmarkLogger:
    def __init__(self, model_name, file_path, delete_existing=True):
        if os.path.isfile(file_path):
            if delete_existing:
                print("{} already exists, deleting it...".format(file_path))
                os.remove(file_path)
            else:
                nb = get_trailing_number(file_path)
                nb = nb + 1 if nb else 1
                file_path = file_path + str(nb)
        self.file = open(file_path, 'w')
        self.file.write(f'====== {model_name} ======\n')
        self.file.flush()

    def close(self):
        self.file.close()

    def write_section_size(self):
        self.file.write("\nModels size (KB):\n")

    def write_section_accuracy(self):
        self.file.write("\nAccuracy:\n")

    def write_section_runtime(self):
        self.file.write("\nRuntime:")

    def write_section_empty_run(self, sleep_interval, run_interval):
        self.file.write(f"\nEmpty-run (sleep {sleep_interval} secs, run {run_interval} secs):")

    def write_model_size(self, model_description, size):
        self.file.write(f"  -{model_description}: {size}\n")
        self.file.flush()

    def write_model_accuracy(self, model_description, accuracy):
        self.file.write(f"  -{model_description}: {accuracy}\n")
        self.file.flush()

    def write_model_runtimes(self, model_description, runtimes):
        self.file.write(f"\n  -{model_description}:\n")
        for bs, runtime in runtimes:
            self.file.write(f"{bs_description(bs)} min: {runtime[0]:<8.3f} max: {runtime[1]:<8.3f} "
                            f"median: {runtime[2]:<8.3f}(ms)\n")
        self.file.flush()

    def write_model_empty_run(self, model_description):
        self.file.write(f"\n  -{model_description}:\n")
        self.file.flush()

    def write_model_sleep(self, batch_size):
        t = re.search("..:..:..", time.asctime()).group()
        self.file.write(f"{bs_description(batch_size)} {t: <12}  sleep\n")
        self.file.flush()

    def write_model_start_run(self, batch_size):
        t = re.search("..:..:..", time.asctime()).group()
        self.file.write(f"{bs_description(batch_size)} {t: <12}  start run\n")
        self.file.flush()

    def write_model_end_run(self, batch_size):
        t = re.search("..:..:..", time.asctime()).group()
        self.file.write(f"{bs_description(batch_size)} {t: <12}  end run\n")
        self.file.flush()

