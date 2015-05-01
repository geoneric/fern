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


namespace fern {

class Messages:
    private std::map<MessageId, std::string>
{

public:

                   Messages            ();

                   Messages            (Messages const&)=delete;

    Messages&      operator=           (Messages const&)=delete;

                   Messages            (Messages const&&)=delete;

    Messages&      operator=           (Messages const&&)=delete;

    std::string const&
                   operator[]          (MessageId message_id) const;

    template<
        class T1>
    std::string    format_message      (MessageId message_id,
                                        T1 const& argument1) const;

    template<
        class T1,
        class T2>
    std::string    format_message      (MessageId message_id,
                                        T1 const& argument1,
                                        T2 const& argument2) const;

    template<
        class T1,
        class T2,
        class T3>
    std::string    format_message      (MessageId message_id,
                                        T1 const& argument1,
                                        T2 const& argument2,
                                        T3 const& argument3) const;

    template<
        class T1,
        class T2,
        class T3,
        class T4>
    std::string    format_message      (MessageId message_id,
                                        T1 const& argument1,
                                        T2 const& argument2,
                                        T3 const& argument3,
                                        T4 const& argument4) const;

private:

};


template<
    class T1>
inline std::string Messages::format_message(
    MessageId message_id,
    T1 const& argument1) const
{
    std::string const& format_string(this->operator[](message_id));
    return std::string((boost::format(format_string)
        % argument1
    ).str());
}


template<
    class T1,
    class T2>
inline std::string Messages::format_message(
    MessageId message_id,
    T1 const& argument1,
    T2 const& argument2) const
{
    std::string const& format_string(this->operator[](message_id));
    return (boost::format(format_string)
        % argument1
        % argument2
    ).str();
}


template<
    class T1,
    class T2,
    class T3>
inline std::string Messages::format_message(
    MessageId message_id,
    T1 const& argument1,
    T2 const& argument2,
    T3 const& argument3) const
{
    std::string const& format_string(this->operator[](message_id));
    return (boost::format(format_string)
        % argument1
        % argument2
        % argument3
    ).str();
}


template<
    class T1,
    class T2,
    class T3,
    class T4>
inline std::string Messages::format_message(
    MessageId message_id,
    T1 const& argument1,
    T2 const& argument2,
    T3 const& argument3,
    T4 const& argument4) const
{
    std::string const& format_string(this->operator[](message_id));
    return (boost::format(format_string)
        % argument1
        % argument2
        % argument3
        % argument4
    ).str();
}

} // namespace fern
