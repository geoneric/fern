#pragma once
#include <future>
#include <thread>
#include <vector>
#include "fern/core/join_threads.h"
#include "fern/core/memory.h"
#include "fern/core/thread_safe_queue.h"


namespace fern {

class FunctionWrapper
{

public:

                   FunctionWrapper     ()=default;

                   FunctionWrapper     (FunctionWrapper&& other);

                   FunctionWrapper     (FunctionWrapper const& other)=delete;

    template<
        class Function>
                   FunctionWrapper     (Function&& function);

    FunctionWrapper& operator=         (FunctionWrapper&& other);

    FunctionWrapper& operator=         (FunctionWrapper const& other)=delete;

    void           operator()          ();

private:

    struct Concept
    {

        virtual ~Concept()
        {
        }

        virtual void call()=0;

    };

    template<
        class Function>
    struct Model:
        public Concept
    {

        Model(
            Function&& function)

            : _function(std::move(function))

        {
        }

        void call()
        {
            _function();
        }

        Function   _function;

    };

    std::unique_ptr<Concept> _concept;

};


template<
    class Function>
inline FunctionWrapper::FunctionWrapper(
    Function&& function)

    // : _concept(std::make_unique<Model<Function>>(std::move(function)))

    : _concept(new Model<Function>(std::move(function)))

{
}


//! Thread pool.
/*!
  \warning   Don't create a thread pool if you don't have anything useful
             for the threads to do. They will continuously yield and will
             occupy the machine by doing nothing.
  \sa        .
*/
class ThreadPool
{

public:

                   ThreadPool          (size_t nr_threads);

                   ~ThreadPool         ();

    template<
        class Function>
    std::future<typename std::result_of<Function()>::type>
                   submit              (Function function);

    size_t         size                () const;

private:

    std::atomic_bool _done;

    ThreadSafeQueue<FunctionWrapper> _work_queue;

    std::vector<std::thread> _threads;

    JoinThreads    _joiner;

    void           worker_thread       ();

};


//! Wrap the function or callable object in a task, push it on the work queue and return its future.
/*!
  \tparam    Function Type of function or callable object.
  \param     function Function or callable object to pass to thread pool.
  \return    Future holding the return value of the task, allowing the caller
             to wait for the task to complete.
*/
template<
    class Function>
inline std::future<typename std::result_of<Function()>::type>
    ThreadPool::submit(
        Function function)
{
    typedef typename std::result_of<Function()>::type result_type;

    std::packaged_task<result_type()> task(std::move(function));
    std::future<result_type> result(task.get_future());
    _work_queue.push(std::move(task));

    return result;
}

} // namespace fern
