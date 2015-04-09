// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <map>
#include <boost/format.hpp>
#include "fern/core/message_id.h"
#include "fern/core/string.h"


namespace fern {

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

    template<
        class T1,
        class T2,
        class T3>
    String         format_message      (MessageId message_id,
                                        T1 const& argument1,
                                        T2 const& argument2,
                                        T3 const& argument3) const;

    template<
        class T1,
        class T2,
        class T3,
        class T4>
    String         format_message      (MessageId message_id,
                                        T1 const& argument1,
                                        T2 const& argument2,
                                        T3 const& argument3,
                                        T4 const& argument4) const;

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
        % argument2
    ).str());
}


template<
    class T1,
    class T2,
    class T3>
inline String Messages::format_message(
    MessageId message_id,
    T1 const& argument1,
    T2 const& argument2,
    T3 const& argument3) const
{
    String const& format_string(this->operator[](message_id));
    return String((boost::format(format_string.encode_in_utf8())
        % argument1
        % argument2
        % argument3
    ).str());
}


template<
    class T1,
    class T2,
    class T3,
    class T4>
inline String Messages::format_message(
    MessageId message_id,
    T1 const& argument1,
    T2 const& argument2,
    T3 const& argument3,
    T4 const& argument4) const
{
    String const& format_string(this->operator[](message_id));
    return String((boost::format(format_string.encode_in_utf8())
        % argument1
        % argument2
        % argument3
        % argument4
    ).str());
}

} // namespace fern
