# Copy all Python modules to a target directory and create a target that
# other targets can depend on.
COPY_PYTHON_MODULES(
    TARGET fern_test_modules
    TARGET_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/test)
