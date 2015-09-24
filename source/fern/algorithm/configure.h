#pragma once
#include <boost/config.hpp>


#if defined(FERN_ALGORITHM_DYN_LINK)
#    if defined(FERN_ALGORITHM_SOURCE)
#        define FERN_ALGORITHM_DECL BOOST_SYMBOL_EXPORT
#    else
#        define FERN_ALGORITHM_DECL BOOST_SYMBOL_IMPORT
#    endif
#else
#    define FERN_ALGORITHM_DECL
#endif
