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
#include "fern/core/data_name.h"
#include "fern/language/interpreter/data_source.h"


namespace fern {

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

} // namespace fern
