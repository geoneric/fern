// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstring>


namespace fern {
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

    bool           is_initialized      () const;

protected:

                   Client              ();

private:

    //! Number of times Python is initialized without being finalized.
    static unsigned short _count;

};

} // namespace python
} // namespace fern
