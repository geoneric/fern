// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/exception.h"


namespace fern {

Messages Exception::_messages;

Messages const& Exception::messages()
{
    return _messages;
}


// Exception::Exception()
// 
//     : _message_id(MessageId::UNKNOWN_ERROR)
// 
// {
// }


Exception::Exception(
    MessageId message_id)

    : _message_id(message_id)

{
}


String Exception::message() const
{
    return _messages[_message_id];
}

} // namespace fern
