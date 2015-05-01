// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Value
{

public:

protected:

                   Value               ()=default;

                   Value               (Value const&)=delete;

    Value&         operator=           (Value const&)=delete;

                   Value               (Value&&)=delete;

    Value&         operator=           (Value&&)=delete;

    virtual        ~Value              ()=default;

private:

};

} // namespace language
} // namespace fern
