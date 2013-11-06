#pragma once
#include <memory>
#include "geoneric/core/data_name.h"
#include "geoneric/interpreter/data_source.h"


namespace geoneric {

class Dataset;


//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class DatasetSource:
    public DataSource
{

public:

                   DatasetSource      (DataName const& data_name);

                   DatasetSource      (DatasetSource const&)=delete;

    DatasetSource& operator=          (DatasetSource const&)=delete;

                   DatasetSource      (DatasetSource&&)=delete;

    DatasetSource& operator=          (DatasetSource&&)=delete;

                   ~DatasetSource     ()=default;

    ExpressionType const& expression_type() const;

    std::shared_ptr<Argument>
                   read                () const;

private:

    DataName const _data_name;

    std::shared_ptr<Dataset> _dataset;

    ExpressionType _expression_type;

};

} // namespace geoneric
