#pragma once
#include <boost/config.hpp>


#if defined(FERN_CORE_DYN_LINK)
#    if defined(FERN_CORE_SOURCE)
#        define FERN_CORE_DECL BOOST_SYMBOL_EXPORT
#    else
#        define FERN_CORE_DECL BOOST_SYMBOL_IMPORT
#    endif
#else
#    define FERN_CORE_DECL
#endif
