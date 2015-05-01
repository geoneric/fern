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
#include <vector>


class GDALDriverManager;
class GDALDriver;


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GDALClient
{

public:

protected:

                   GDALClient          ();

                   GDALClient          (GDALClient const&)=delete;

    GDALClient&    operator=           (GDALClient const&)=delete;

                   GDALClient          (GDALClient&&)=delete;

    GDALClient&    operator=           (GDALClient&&)=delete;

    virtual        ~GDALClient         ();

private:

    //! Number of times an instance is created.
    static size_t  _count;

    static std::vector<std::string>
                   _names_of_drivers_to_skip;

    static void    insert_names_of_drivers_to_skip
                                       ();

    static void    erase_driver        (GDALDriverManager& manager,
                                        std::string const& name,
                                        GDALDriver* driver);

    static void    erase_drivers_to_skip
                                       ();

    static size_t  nr_drivers_to_skip  ();

    static bool    skip_driver         (std::string const& name);

    static void    register_all_drivers();

    static void    deregister_all_drivers();

};

} // namespace language
} // namespace fern
