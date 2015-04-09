// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <cstdlib>


int main(
    int /* argc */,
    char** /* argv */)
{
    int status = EXIT_FAILURE;

    try {
        // TODO Think of tests to do.

        // - Time handwritten sum with 1D array.
        // - Time fern's sum with 1D array with same number of cells.
        //   - Default policies.
        // -> Performance must be similar.

        // - Time sum with 1D array.
        // - Time sum with 2D array with same number of cells.
        // - Default policies.
        // -> Performance must be similar.

        // - Time handwritten sum with 1D array.
        // - Time fern's sum with 2D array with same number of cells.
        // - Add a performance overflow at the start.
        // - Fern's sum must bail out immediately.
        // -> Performance must be similar.

        // - Time fern's sum with 2D array.
        // - Both serial and concurrent execution.
        // - Concurrent must be much faster. Almost nr_cores times faster.

        status = EXIT_SUCCESS;
    }
    catch(...) {
    }

    return status;
}
