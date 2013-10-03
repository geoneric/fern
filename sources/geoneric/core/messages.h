#pragma once
#include <map>
#include <boost/format.hpp>
#include "geoneric/core/message_id.h"
#include "geoneric/core/string.h"


namespace geoneric {

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

    template<
        class T1>
    String         format_message      (MessageId message_id,
                                        T1 const& argument1) const;

    template<
        class T1,
        class T2>
    String         format_message      (MessageId message_id,
                                        T1 const& argument1,
                                        T2 const& argument2) const;

private:

};


template<
    class T1>
inline String Messages::format_message(
    MessageId message_id,
    T1 const& argument1) const
{
    String const& format_string(this->operator[](message_id));
    return String((boost::format(format_string.encode_in_utf8())
        % argument1
    ).str());
}


template<
    class T1,
    class T2>
inline String Messages::format_message(
    MessageId message_id,
    T1 const& argument1,
    T2 const& argument2) const
{
    String const& format_string(this->operator[](message_id));
    return String((boost::format(format_string.encode_in_utf8())
        % argument1
        % argument1
        % argument2
    ).str());
}

} // namespace geoneric
