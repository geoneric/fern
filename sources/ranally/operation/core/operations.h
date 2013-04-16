#pragma once
#include <map>
#include <memory>
#include <boost/format.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/iterator.hpp>
#include "ranally/core/string.h"
#include "ranally/operation/core/operation.h"


namespace ranally {

//! Class for storing information about individual operations.
/*!
*/
class Operations
{

    friend class OperationsTest;

public:

    template<class Range>
                   Operations          (Range const& operations);

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
    typedef typename boost::range_iterator<Range const>::type Iterator;
    Iterator end = boost::end(operations);

    for(Iterator it = boost::begin(operations); it != end; ++it) {
        OperationPtr const& operation(*it);
        assert(_operations.find(operation->name()) == _operations.end());
        _operations[operation->name()] = operation;
    }
}


typedef std::shared_ptr<Operations> OperationsPtr;

} // namespace ranally
