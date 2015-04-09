// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <vector>
#include "fern/interpreter/data_sources.h"
#include "fern/interpreter/data_syncs.h"
#include "fern/compiler/data_description.h"


namespace fern {

class Module
{

public:

    std::vector<DataDescription> const&
                   arguments           () const;

    std::vector<DataDescription> const&
                   results             () const;

    virtual void   run                 (std::vector<std::shared_ptr<
                                            DataSource>> const& data_sources,
                                        std::vector<std::shared_ptr<
                                            DataSync>> const& data_syncs) const=0;

protected:

                   Module              (std::vector<DataDescription> const&
                                            arguments,
                                        std::vector<DataDescription> const&
                                            results);

    virtual        ~Module             ()=default;

                   Module              (Module&&)=delete;

    Module&        operator=           (Module&&)=delete;

                   Module              (Module const&)=delete;

    Module&        operator=           (Module const&)=delete;

    void           check_sources_and_syncs(
                                        std::vector<std::shared_ptr<
                                            DataSource>> const& data_sources,
                                        std::vector<std::shared_ptr<
                                            DataSync>> const& data_syncs) const;

private:

    std::vector<DataDescription> const _arguments;

    std::vector<DataDescription> const _results;

};

} // namespace fern
