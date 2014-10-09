from .process import ChangeDirectory, run_command


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
