#include <vector>


std::vector<int>::iterator max_element(
    std::vector<int>::iterator first,
    std::vector<int>::iterator last)
{
    std::vector<int>::iterator it = first;

    for(++first; first != last; ++first) {
        if(*first > *it) {
            it = first;
        }
    }

    return it;
}
