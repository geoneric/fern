#pragma once
#include <map>
#include "ranally/core/message_id.h"
#include "ranally/core/string.h"


namespace ranally {

class Messages:
    private std::map<MessageId, String>
{

public:

                   Messages            ();

                   Messages            (Messages const&)=delete;

    Messages&      operator=           (Messages const&)=delete;

                   Messages            (Messages const&&)=delete;

    Messages&      operator=           (Messages const&&)=delete;

    String const&  operator[]          (MessageId message_id) const;

private:

};

} // namespace ranally
