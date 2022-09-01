// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/exception.h"


namespace fern {


/*!
    @brief      Exception for I/O related errors.
*/
class IOError:
    public Exception
{
public:

                   IOError             (std::string const& source_name,
                                        std::string const& message);

                   IOError             (std::string const& source_name,
                                        int errno_);

                   IOError             (IOError const&)=default;

    IOError&       operator=           (IOError const&)=default;

                   IOError             (IOError&&)=default;

    IOError&       operator=           (IOError&&)=default;

    std::string    message             () const override;

private:

    std::string    _source_name;

    std::string    _message;

};

} // namespace fern
