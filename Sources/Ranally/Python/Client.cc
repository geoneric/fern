#include <Python.h>
#include "Ranally/Python/Client.h"


namespace ranally {
namespace python {

unsigned short Client::_count = 0;


Client::Client()

    : _initialized(false)

{
    // Python docs: There is no return value; it is a fatal error if the
    // initialization fails.
    Py_Initialize();
    _initialized = true;
    ++_count;
}


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


bool Client::isInitialized() const
{
    return _initialized;
}

} // namespace python
} // namespace ranally
