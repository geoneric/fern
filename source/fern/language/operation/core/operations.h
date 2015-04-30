// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <map>
#include <memory>
#include <boost/format.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/iterator.hpp>
#include "fern/core/string.h"
#include "fern/language/operation/core/operation.h"


namespace fern {

//! Class for storing information about individual operations.
/*!
*/
class Operations
{

    friend class OperationsTest;

public:

    template<class Range>
                   Operations          (Range const& operations);

                   Operations          (std::initializer_list<OperationPtr>
                                            values);

                   ~Operations         ()=default;

                   Operations          (Operations&&)=delete;

    Operations&    operator=           (Operations&&)=delete;

                   Operations          (Operations const&)=delete;

    Operations&    operator=           (Operations const&)=delete;

    bool           empty               () const;

    size_t         size                () const;

    bool           has_operation       (String const& name) const;

    OperationPtr const& operation      (String const& name) const;

private:

    //! Collection of operations, by name.
    std::map<String, OperationPtr> _operations;

};


//! Construct an Operations instance.
/*!
  \tparam    Range Class of collection containing OperationPtr instances.
  \param     operations Collection of OperationPtr instances.
*/
template<
    class Range>
inline Operations::Operations(
    Range const& operations)
{
    using Iterator = typename boost::range_iterator<Range const>::type;
    Iterator end = boost::end(operations);

    for(Iterator it = boost::begin(operations); it != end; ++it) {
        OperationPtr const& operation(*it);
        assert(_operations.find(operation->name()) == _operations.end());
        _operations[operation->name()] = operation;
    }
}


using OperationsPtr = std::shared_ptr<Operations>;

} // namespace fern
