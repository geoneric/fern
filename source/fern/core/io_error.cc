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
    String const& source_name,
    String const& message)

    : Exception(MessageId::IO_ERROR),
      _source_name(source_name),
      _message(message)

{
}


IOError::IOError(
    String const& source_name,
    int errno_)

    : Exception(MessageId::IO_ERROR),
      _source_name(source_name),
      _message(std::strerror(errno_))

{
}


String IOError::message() const
{
    return boost::format(Exception::message().encode_in_utf8())
        % _source_name.encode_in_utf8()
        % _message.encode_in_utf8()
        ;
}

} // namespace fern
