#pragma once
#include <cstring>


namespace ranally {
namespace python {

//! Utility class for clients that use the Python library.
/*!
  Creating a Client instance will initialize the Python library. Once
  the object goes out of scope the matching finalize will be called.

  It is possible to instantiate more than one Client objects in the
  same application.
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

    //! Number of times Python is initialized without being finalized.
    static unsigned short _count;

    //! Whether initialization of the Python library succeeded.
    bool             _initialized;

};

} // namespace python
} // namespace ranally
