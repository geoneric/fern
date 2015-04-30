// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/expression_type.h"
#include "fern/language/operation/core/argument.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        DataSource
*/
class DataSync
{

public:

    virtual void   write               (Argument const& argument)=0;

protected:

                   DataSync            ()=default;

                   DataSync            (DataSync const&)=delete;

    DataSync&      operator=           (DataSync const&)=delete;

                   DataSync            (DataSync&&)=delete;

    DataSync&      operator=           (DataSync&&)=delete;

    virtual        ~DataSync           ()=default;

private:

};

} // namespace language
} // namespace fern
