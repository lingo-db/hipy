#ifndef BUILTIN_COMMONS_H
#define BUILTIN_COMMONS_H
#include<memory>
#include<vector>
#include <cstdint>
namespace builtin {
    template<class ElT>
    using List=std::shared_ptr<std::vector<ElT>>;
    template<typename FuncT, typename ClosureT>
    class bound_fn {
    public:
        bound_fn(FuncT func, ClosureT closure) : func_(func), closure_(closure) {
        }

        template<typename... Args>
        auto operator()(Args&&... args) const {
            return call(std::index_sequence_for<Args...>(), std::forward<Args>(args)...);
        }

    private:
        template<std::size_t... Is, typename... Args>
        auto call(std::index_sequence<Is...>, Args&&... args) const {
            return func_(std::forward<Args>(args)..., closure_);
        }

        FuncT func_;
        ClosureT closure_;
    };
}
#endif //BUILTIN_COMMONS_H
