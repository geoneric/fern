// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
// #include <condition_variable>
// #include <memory>
#include <mutex>
#include <queue>


namespace fern {

template<
    class T>
class ThreadSafeQueue
{

public:

                   ThreadSafeQueue     ();

    void           push                (T value);

    // void           wait_and_pop        (T& value);

    // std::shared_ptr<T> wait_and_pop    ();

    bool           try_pop             (T& value);

    // std::shared_ptr<T> try_pop         ();

    bool           empty               () const;

private:

    mutable std::mutex _mutex;

    std::queue<T> _data_queue;

    // std::condition_variable _data_condition;

};


template<
    class T>
ThreadSafeQueue<T>::ThreadSafeQueue()
{
}


template<
    class T>
void ThreadSafeQueue<T>::push(
    T value)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _data_queue.push(std::move(value));
    // _data_condition.notify_one();
}


template<
    class T>
bool ThreadSafeQueue<T>::try_pop(
    T& value)
{
    std::lock_guard<std::mutex> lock(_mutex);

    if(_data_queue.empty()) {
        return false;
    }

    value = std::move(_data_queue.front());
    _data_queue.pop();

    return true;
}


template<
    class T>
bool ThreadSafeQueue<T>::empty() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _data_queue.empty();
}

} // namespace fern
