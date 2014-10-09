import os
import subprocess


class ChangeDirectory:
    """
    Context manager for changing the current working directory.
    """

    def __init__(self,
            new_path):
        self.new_path = new_path

    def __enter__(self):
        self.orinal_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self,
            exception_type,
            exception_value,
            traceback):
        os.chdir(self.orinal_path)


def run_command(
        command):
    subprocess.check_call(command.split())
