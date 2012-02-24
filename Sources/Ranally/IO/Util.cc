#include "Ranally/IO/Util.h"



namespace ranally {
namespace io {

void import(
  UnicodeString const& inputDataSetName,
  UnicodeString const& outputDataSetName)
{
  // Open input data set.
  // Open output data set.
  // Loop over stuff in input data set and write it to output data set, while
  // printing progress.
  // Don't hog the system, just read/write sequentially, chunk by chunk.
}

} // namespace io
} // namespace ranally

