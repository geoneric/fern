#include "Ranally/Language/Operation/Requirements.h"


namespace ranally {

Requirements::Requirements()
{
}


Requirements::~Requirements()
{
}


std::vector<Argument> const& Requirements::arguments()
{
    return _arguments;
}


std::vector<Result> const& Requirements::results()
{
    return _results;
}

} // namespace ranally
