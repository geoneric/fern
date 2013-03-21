#pragma once
#include <memory>
#include <vector>
#include "ranally/core/string.h"
#include "ranally/operation/parameter.h"
#include "ranally/operation/result.h"
#include "ranally/operation/result_type.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
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

    String const&  name                () const;

    String const&  description         () const;

    size_t         arity               () const;

    std::vector<Parameter> const& parameters() const;

    std::vector<Result> const& results () const;

    ResultType     result_type         (
                        size_t index,
                        std::vector<ResultType> const& argument_types) const;

private:

    String         _name;

    String         _description;

    std::vector<Parameter> _parameters;

    std::vector<Result> _results;

};


typedef std::shared_ptr<Operation> OperationPtr;

} // namespace ranally
