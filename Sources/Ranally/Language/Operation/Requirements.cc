#include "Ranally/Language/Operation/Requirements.h"



namespace ranally {
namespace language {
namespace operation {

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

} // namespace operation
} // namespace language
} // namespace ranally

