#include <cassert>
#include <malloc.h>
#include <iostream>
#include <memory>
#include <string>


// This program can be used to test whether page faults occur.
int main(
    int argc,
    char** argv)
{
    assert(argc == 2);
    int status = EXIT_FAILURE;

    // Turn off malloc trimming.
    mallopt(M_TRIM_THRESHOLD, -1);
    // Turn off mmap usage.
    mallopt(M_MMAP_MAX, 0);

    try {
        int const nr_bytes_in_megabyte = 1024 * 1024;
        int const nr_megabytes = std::stoi(argv[1]);
        assert(nr_megabytes > 0);
        int const nr_bytes = nr_megabytes * nr_bytes_in_megabyte;

        while(true) {
            // Allocate the number of bytes, and de-allocate them again.
            std::unique_ptr<int8_t>(new int8_t[nr_bytes]);
        }
    }
    catch(std::exception const& exception) {
        std::cerr << exception.what() << '\n';
        status = EXIT_FAILURE;
    }

    return status;
}
