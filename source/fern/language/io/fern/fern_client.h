// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <string>


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FernClient
{

public:

protected:

                   FernClient          ();

                   FernClient          (FernClient const&)=delete;

    FernClient&    operator=           (FernClient const&)=delete;

                   FernClient          (FernClient&&)=delete;

    FernClient&    operator=           (FernClient&&)=delete;

    virtual        ~FernClient         ();

private:

    //! Number of times an instance is created.
    static size_t  _count;

    static std::string const _driver_name;

    void           register_driver();

    void           deregister_driver();

};

} // namespace language
} // namespace fern
