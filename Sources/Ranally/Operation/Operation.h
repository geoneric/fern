#ifndef INCLUDED_RANALLY_OPERATION_OPERATION
#define INCLUDED_RANALLY_OPERATION_OPERATION

#include <vector>
#include <unicode/unistr.h>
#include <boost/shared_ptr.hpp>



namespace ranally {
namespace operation {

class Parameter;
class Result;

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
                                  UnicodeString const& name,
                                  UnicodeString const& description,
                                  std::vector<Parameter> const& parameters,
                                  std::vector<Result> const& results);

                   Operation           (Operation const& other);

  Operation&       operator=           (Operation const& other);

                   ~Operation          ();

  UnicodeString const& name            () const;

  UnicodeString const& description     () const;

  std::vector<Parameter> const& parameters() const;

  std::vector<Result> const& results   () const;

private:

  UnicodeString    _name;

  UnicodeString    _description;

  std::vector<Parameter> _parameters;

  std::vector<Result> _results;

};



typedef boost::shared_ptr<Operation> OperationPtr;

} // namespace operation
} // namespace ranally

#endif