#ifndef INCLUDED_RANALLY_IO_UTIL
#define INCLUDED_RANALLY_IO_UTIL

#include <unicode/unistr.h>


namespace ranally {
namespace io {

void               import              (UnicodeString const& inputDataSetName,
                                        UnicodeString const& outputDataSetName);

} // namespace io
} // namespace ranally

#endif
