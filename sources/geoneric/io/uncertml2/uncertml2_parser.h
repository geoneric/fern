#pragma once
#include <memory>
#include "geoneric/core/string.h"


namespace geoneric {

class Uncertainty;

//! TODO
/*!
   TODO
*/
class UncertML2Parser
{

public:

                   UncertML2Parser     ()=default;

                   ~UncertML2Parser    ()=default;

                   UncertML2Parser     (UncertML2Parser&&)=delete;

    UncertML2Parser& operator=         (UncertML2Parser&&)=delete;

                   UncertML2Parser     (UncertML2Parser const&)=delete;

    UncertML2Parser& operator=         (UncertML2Parser const&)=delete;

    std::shared_ptr<Uncertainty> parse (String const& xml) const;

private:

    std::shared_ptr<Uncertainty> parse (std::istream& stream) const;

};

} // namespace geoneric
