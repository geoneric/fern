// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/io/core/driver.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FernDriver:
    public Driver
{

public:

                   FernDriver          ();

                   FernDriver          (FernDriver const&)=delete;

    FernDriver&    operator=           (FernDriver const&)=delete;

                   FernDriver          (FernDriver&&)=delete;

    FernDriver&    operator=           (FernDriver&&)=delete;

                   ~FernDriver         ()=default;

    bool           can_open            (std::string const& name,
                                        OpenMode open_mode);

    // ExpressionType expression_type     (DataName const& data_name);

    std::shared_ptr<Dataset> open      (std::string const& name,
                                        OpenMode open_mode);

private:

    bool           can_open_for_read   (std::string const& name);

    bool           can_open_for_overwrite(
                                        std::string const& name);

    bool           can_open_for_update (std::string const& name);

};

} // namespace language
} // namespace fern
