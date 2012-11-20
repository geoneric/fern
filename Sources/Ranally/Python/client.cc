#include <Python.h>
#include "Ranally/Python/client.h"


namespace ranally {
namespace python {

unsigned short Client::_count = 0;


//! Construct a Client instance.
/*!
  The Python API will be initialized.

  From the Python docs: There is no return value; it is a fatal error if the
  initialization fails.
*/
Client::Client()

    : _initialized(false)

{
    Py_Initialize();
    _initialized = true;
    ++_count;
}


//! Destruct a Client instance.
/*!
*/
Client::~Client()
{
    if(_initialized) {
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

  Since there currently is not a way to check whether initialization of the
  Python API failed (see Client()), this method always returns true.
*/
bool Client::isInitialized() const
{
    return _initialized;
}

} // namespace python
} // namespace ranally
