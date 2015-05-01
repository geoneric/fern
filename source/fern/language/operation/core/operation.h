// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "fern/core/expression_type.h"
#include "fern/language/operation/core/argument.h"
#include "fern/language/operation/core/parameter.h"
#include "fern/language/operation/core/result.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .

  TODO Make this an abstract base class for all operations.
*/
class Operation
{

    friend class OperationTest;

public:

                   Operation           (
                                  std::string const& name,
                                  std::string const& description,
                                  std::vector<Parameter> const& parameters,
                                  std::vector<Result> const& results);

                   ~Operation          ()=default;

                   Operation           (Operation&& other);

    Operation&     operator=           (Operation&& other);

                   Operation           (Operation const& other);

    Operation&     operator=           (Operation const& other);

    // std::string         xml                 () const;

    std::string const&
                   name                () const;

    std::string const&
                   description         () const;

    size_t         arity               () const;

    std::vector<Parameter> const& parameters() const;

    std::vector<Result> const& results () const;

    virtual ExpressionType expression_type(
                                        size_t index,
                   std::vector<ExpressionType> const& argument_types) const;

    virtual std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const=0;

private:

    std::string    _name;

    std::string    _description;

    std::vector<Parameter> _parameters;

    std::vector<Result> _results;

};


using OperationPtr = std::shared_ptr<Operation>;

} // namespace language
} // namespace fern
