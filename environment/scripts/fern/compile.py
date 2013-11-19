import shutil
import tempfile
import fern.cmake
import fern.process


def compile_model_to_cpp(
        model_pathname,
        output_directory_pathname,
        dump_driver=False,
        dump_cmake=False):
    command = "fern compile c++ {} {} {} {}".format(
        "--dump_driver" if dump_driver else "",
        "--dump_cmake" if dump_cmake else "",
        model_pathname, output_directory_pathname)
    fern.process.run_command(command)


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
""".format(source_directory_pathname, binary_directory_pathname,
            install_directory_pathname))
        compile_model_to_cpp(model_pathname, source_directory_pathname,
            dump_driver=True, dump_cmake=True)
        fern.cmake.configure_cmake_project(source_directory_pathname,
            binary_directory_pathname, install_directory_pathname)
        fern.cmake.build_cmake_project(binary_directory_pathname)
        fern.cmake.install_cmake_project(binary_directory_pathname)
    finally:
        print("TODO Not removing temp stuff...")
        # if source_directory_pathname:
        #     shutil.rmtree(source_directory_pathname)
        # if binary_directory_pathname:
        #     shutil.rmtree(binary_directory_pathname)
