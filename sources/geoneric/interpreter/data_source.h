#pragma once
#include "geoneric/core/expression_type.h"
#include "geoneric/operation/core/argument.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class DataSource
{

public:

    virtual ExpressionType const&
                   expression_type     () const=0;

    virtual std::shared_ptr<Argument>
                   read                () const=0;

protected:

                   DataSource          ()=default;

                   DataSource          (DataSource const&)=delete;

    DataSource&    operator=           (DataSource const&)=delete;

                   DataSource          (DataSource&&)=delete;

    DataSource&    operator=           (DataSource&&)=delete;

    virtual        ~DataSource         ()=default;

private:

};

} // namespace geoneric
