#pragma once
#include <vector>
#include "fern/interpreter/data_source.h"
#include "fern/compiler/data_description.h"


namespace fern {

class Module
{

public:

    virtual void   run                 ();

protected:

                   Module              (std::vector<DataDescription> const&
                                            arguments,
                                        std::vector<std::shared_ptr<
                                            DataSource>> const& data_sources);

                   Module              (std::vector<DataDescription> const&
                                            arguments,
                                        int argc,
                                        char** argv);

    virtual        ~Module             ()=default;

                   Module              (Module&&)=delete;

    Module&        operator=           (Module&&)=delete;

                   Module              (Module const&)=delete;

    Module&        operator=           (Module const&)=delete;

private:

    std::vector<DataDescription> const _arguments;

    std::vector<std::shared_ptr<DataSource>> _data_sources;

};

} // namespace fern
