#pragma once
#include <map>
#include <boost/format.hpp>
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

    template<class T1>
    String         format_message      (MessageId message_id,
                                        T1 const& argument1) const;

private:

};


template<class T1>
inline String Messages::format_message(
    MessageId message_id,
    T1 const& argument1) const
{
    String const& format_string(this->operator[](message_id));
    return String((boost::format(format_string.encode_in_utf8())
        % argument1
    ).str());
}

} // namespace ranally
