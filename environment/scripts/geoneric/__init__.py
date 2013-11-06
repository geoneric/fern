import functools
import os
import shutil
import sys
import subprocess
import tempfile
import traceback


def checked_call(
        function):
    @functools.wraps(function)
    def wrapper(
            *args,
            **kwargs):
        result = 0
        try:
            result = function(*args, **kwargs)
        except:
            traceback.print_exc(file=sys.stderr)
            result = 1
        return 0 if result is None else result
    return wrapper


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


def compile_model_to_cpp(
        model_pathname,
        output_directory_pathname,
        dump_driver=False,
        dump_cmake=False):
    command = "geoneric compile c++ {} {} {} {}".format(
        "--dump_driver" if dump_driver else "",
        "--dump_cmake" if dump_cmake else "",
        model_pathname, output_directory_pathname)
    run_command(command)


def configure_cmake_project(
        source_directory_pathname,
        binary_directory_pathname,
        install_directory_pathname):
    with ChangeDirectory(binary_directory_pathname):
        command = "cmake -DCMAKE_BUILD_TYPE=Release " \
            "-DCMAKE_INSTALL_PREFIX={} " \
            "{}".format(install_directory_pathname,
                source_directory_pathname)
        run_command(command)


def build_cmake_project(
        binary_directory_pathname):
    command = "cmake --build {} --target all --config Release".format(
        binary_directory_pathname)
    run_command(command)


def install_cmake_project(
        binary_directory_pathname):
    command = "cmake --build {} --target install --config Release".format(
        binary_directory_pathname)
    run_command(command)


def model_to_executable(
        model_pathname,
        install_directory_pathname):
    """
    Do whatever it takes to turn the model passed in to a native executable.

    The only end result is the executable.
    """
    source_directory_pathname, binary_directory_pathname = None, None

    try:
        source_directory_pathname = tempfile.mkdtemp()
        binary_directory_pathname = tempfile.mkdtemp()
        print("""
source : {}
binary : {}
install: {}
""".format(source_directory_pathname, binary_directory_pathname, install_directory_pathname))
        compile_model_to_cpp(model_pathname, source_directory_pathname,
            dump_driver=True, dump_cmake=True)
        configure_cmake_project(source_directory_pathname,
            binary_directory_pathname, install_directory_pathname)
        build_cmake_project(binary_directory_pathname)
        install_cmake_project(binary_directory_pathname)
    finally:
        pass
        # if source_directory_pathname:
        #     shutil.rmtree(source_directory_pathname)
        # if binary_directory_pathname:
        #     shutil.rmtree(binary_directory_pathname)
