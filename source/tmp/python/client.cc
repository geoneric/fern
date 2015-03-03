#include <Python.h>
#include "fern/python/client.h"


namespace fern {
namespace python {

unsigned short Client::_count = 0;


//! Construct a Client instance.
/*!
  The Python API will be initialized.

  From the Python docs: There is no return value; it is a fatal error if the
  initialization fails.
*/
Client::Client()
{
    Py_Initialize();
    ++_count;
}


//! Destruct a Client instance.
/*!
*/
Client::~Client()
{
    if(is_initialized()) {
        assert(_count > 0);
        // Python docs: There is no return value; errors during finalization are
        // ignored.
        Py_Finalize();
        --_count;
    }
}


//! Return whether the Python API is initialized.
/*!
  \return    true or false.
*/
bool Client::is_initialized() const
{
    return Py_IsInitialized() != 0;
}

} // namespace python
} // namespace fern
