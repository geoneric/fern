#pragma once
#include <boost/noncopyable.hpp>
#include "Ranally/Language/Operation/DataTypes.h"
#include "Ranally/Language/Operation/ValueTypes.h"


namespace ranally {
namespace language {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Parameter:
  private boost::noncopyable
{

  friend class ParameterTest;

public:

                   Parameter           ();

  virtual          ~Parameter          ();

  ValueTypes       valueTypes          () const;

  DataTypes        dataTypes           () const;

protected:

private:

  ValueTypes       _valueTypes;

  DataTypes        _dataTypes;

};

} // namespace operation
} // namespace language
} // namespace ranally
