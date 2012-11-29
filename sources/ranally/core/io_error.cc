#include "ranally/core/io_error.h"


namespace ranally {

IOError::IOError(
    String const& filename,
    String const& message)

    : Exception(MessageId::IO_ERROR),
      _filename(filename),
      _message(message)

{
}


IOError::IOError(
    String const& filename,
    int errno_)

    : Exception(MessageId::IO_ERROR),
      _filename(filename),
      _message(std::strerror(errno_))

{
}


IOError::~IOError() noexcept(true) =default;


String IOError::message() const
{
    return boost::format(Exception::message().encode_in_utf8())
        % _filename.encode_in_utf8()
        % _message.encode_in_utf8()
        ;
}

} // namespace ranally
