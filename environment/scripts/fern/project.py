import glob
import os
import fern.process


def rename_project(
    old_name,
    new_name):
    """
    Do whatever is needed to rename the project from *old_name* to *new_name*.

    The goal is that everything compiles again afterwards.
    """
    # Loop over all source directories and do some text replacements.
    commands = [
        "perl -0777 -pi -e 's/include \"{}/include \"{}/g' *.h *.cc"
            .format(old_name, new_name)
    ]

    for root_directory_pathname, _, _ in os.walk(os.path.join(
            os.environ["FERN"], "sources", "fern")):

        with fern.process.ChangeDirectory(root_directory_pathname):
            # for command in commands:
            #     fern.process.run_command(command)

            header_filenames = glob.glob("*.h")
            module_filenames = glob.glob("*.cc")
            source_filenames = header_filenames + module_filenames

            for source_filename in source_filenames:
                source = file(source_filename).read()
                source = source \
                    .replace(
                        "include \"{}".format(old_name),
                        "include \"{}".format(new_name)) \
                    .replace(
                        "namespace {}".format(old_name),
                        "namespace {}".format(new_name)) \
                    .replace(
                        "{}::".format(old_name),
                        "{}::".format(new_name))
                file(source_filename, "w").write(source)

            test_module_filenames = glob.glob("*_test.cc")
            for test_module_filename in test_module_filenames:
                source = file(test_module_filename).read()
                source = source \
                    .replace(
                        "BOOST_TEST_MODULE {}".format(old_name),
                        "BOOST_TEST_MODULE {}".format(new_name))
                file(test_module_filename, "w").write(source)

            cmake_lists_filenames = glob.glob("CMakeLists.txt")
            for cmake_lists_filename in cmake_lists_filenames:
                source = file(cmake_lists_filename).read()
                source = source \
                    .replace(
                        "{}_".format(old_name),
                        "{}_".format(new_name))
                file(cmake_lists_filename, "w").write(source)
