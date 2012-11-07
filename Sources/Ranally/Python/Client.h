#pragma once
#include <cstring>


namespace ranally {
namespace python {

//! Utility class for clients that use the Python library.
/*!
  Creating a Client object will initialize the Python library. Once
  the object goes out of scope the matching terminate will be called.

  It is possible to instantiate more than one Client objects in the
  same application.

  Unfortunately, not all memory will be freed.
  - http://bugs.python.org/issue1445210
*/
class Client
{

    friend class PythonClientTest;

public:

                   Client              (Client&&)=delete;

    Client&        operator=           (Client&&)=delete;

                   Client              (Client const&)=delete;

    Client&        operator=           (Client const&)=delete;

    virtual        ~Client             ();

    bool           isInitialized       () const;

protected:

                   Client              ();

private:

    //! Number of times Python is initialized without being terminated.
    static unsigned short _count;

    //! Whether initialization of the Python library succeeded.
    bool             _initialized;

};

} // namespace python
} // namespace ranally
