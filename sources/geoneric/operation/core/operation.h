#pragma once
#include <memory>
#include <vector>
#include "geoneric/core/string.h"
#include "geoneric/operation/core/argument.h"
#include "geoneric/operation/core/expression_type.h"
#include "geoneric/operation/core/parameter.h"
#include "geoneric/operation/core/result.h"


namespace geoneric {

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
                                  String const& name,
                                  String const& description,
                                  std::vector<Parameter> const& parameters,
                                  std::vector<Result> const& results);

                   ~Operation          ()=default;

                   Operation           (Operation&& other);

    Operation&     operator=           (Operation&& other);

                   Operation           (Operation const& other);

    Operation&     operator=           (Operation const& other);

    // String         xml                 () const;

    String const&  name                () const;

    String const&  description         () const;

    size_t         arity               () const;

    std::vector<Parameter> const& parameters() const;

    std::vector<Result> const& results () const;

    virtual ExpressionType expression_type(
                                        size_t index,
                   std::vector<ExpressionType> const& argument_types) const=0;

    virtual std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const=0;

private:

    String         _name;

    String         _description;

    std::vector<Parameter> _parameters;

    std::vector<Result> _results;

};


typedef std::shared_ptr<Operation> OperationPtr;

} // namespace geoneric
