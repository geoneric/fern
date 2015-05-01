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

class DataDescription
{

public:

                   DataDescription     (std::string const& name);

  std::string      name                () const;

  std::string      description         () const;

private:

  std::string const _name;

  std::string      _description;

};

} // namespace language
} // namespace fern
