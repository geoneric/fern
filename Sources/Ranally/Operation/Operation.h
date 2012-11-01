#pragma once
#include <vector>
#include <boost/shared_ptr.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Operation/Parameter.h"
#include "Ranally/Operation/Result.h"


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

                   Operation           (Operation const& other);

    Operation&     operator=           (Operation const& other);

                   ~Operation          ();

    String const&  name                () const;

    String const&  description         () const;

    std::vector<Parameter> const& parameters() const;

    std::vector<Result> const& results   () const;

private:

    String         _name;

    String         _description;

    std::vector<Parameter> _parameters;

    std::vector<Result> _results;

};


typedef boost::shared_ptr<Operation> OperationPtr;

} // namespace ranally
