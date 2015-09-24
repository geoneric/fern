// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/io_error.h"
#include <cstring>


namespace fern {

IOError::IOError(
    std::string const& source_name,
    std::string const& message)

    : Exception(MessageId::IO_ERROR),
      _source_name(source_name),
      _message(message)

{
}


#if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 4996)
#endif
IOError::IOError(
    std::string const& source_name,
    int errno_)

    : Exception(MessageId::IO_ERROR),
      _source_name(source_name),
      _message(std::strerror(errno_))

{
}
#if defined(_MSC_VER)
#   pragma warning(pop)
#endif


std::string IOError::message() const
{
    return (boost::format(Exception::message())
        % _source_name
        % _message
        ).str();
}

} // namespace fern
