#pragma once
#include <boost/config.hpp>


#if defined(FERN_FEATURE_DYN_LINK)
#    if defined(FERN_FEATURE_SOURCE)
#        define FERN_FEATURE_DECL BOOST_SYMBOL_EXPORT
#    else
#        define FERN_FEATURE_DECL BOOST_SYMBOL_IMPORT
#    endif
#else
#    define FERN_FEATURE_DECL
#endif
