#include "ranally/core/io_error.h"


namespace ranally {

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


IOError::~IOError() noexcept(true) =default;


String IOError::message() const
{
    return boost::format(Exception::message().encode_in_utf8())
        % _source_name.encode_in_utf8()
        % _message.encode_in_utf8()
        ;
}

} // namespace ranally
