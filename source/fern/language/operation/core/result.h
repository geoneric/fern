// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <string>
#include "fern/core/expression_type.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Result
{

    friend class ResultTest;

public:

                   Result              (std::string const& name,
                                        std::string const& description,
                                        ExpressionType const& expression_type);

                   ~Result             ()=default;

                   Result              (Result&& other);

    Result&        operator=           (Result&& other);

                   Result              (Result const& other);

    Result&        operator=           (Result const& other);

    std::string const&
                   name                () const;

    std::string const&
                   description         () const;

    ExpressionType const& expression_type() const;

private:

    std::string    _name;

    std::string    _description;

    ExpressionType _expression_type;

};

} // namespace language
} // namespace fern
