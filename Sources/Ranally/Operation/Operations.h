#ifndef INCLUDED_RANALLY_OPERATION_OPERATIONS
#define INCLUDED_RANALLY_OPERATION_OPERATIONS

#include <map>
#include <boost/format.hpp>
#include <boost/noncopyable.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/iterator.hpp>
#include <boost/shared_ptr.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Operation/Operation.h"



namespace ranally {
namespace operation {

//! Class for storing information about individual operations.
/*!
*/
class Operations:
  private boost::noncopyable
{

  friend class OperationsTest;

public:

  template<class Range>
                   Operations          (Range const& operations);

                   ~Operations         ();

  bool             empty               () const;

  size_t           size                () const;

  bool             hasOperation        (String const& name) const;

  OperationPtr const& operation        (String const& name) const;

private:

  //! Collection of operations, by name.
  std::map<String, OperationPtr> _operations;

};



//! Construct an Operations instance.
/*!
  \tparam    Range Class of collection containing OperationPtr instances.
  \param     operations Collection of OperationPtr instances.
  \exception std::runtime_error In case \a operations contains multiple
             instances with the same name.
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

    if(_operations.find(operation->name()) != _operations.end()) {
      throw std::runtime_error((boost::format(
        "operation %1% already present")
        % operation->name().encodeInUTF8()).str().c_str());
    }

    _operations[operation->name()] = operation;
  }
}



typedef boost::shared_ptr<Operations> OperationsPtr;

} // namespace operation
} // namespace ranally

#endif
