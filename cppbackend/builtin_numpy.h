#ifndef BUILTIN_NUMPY_H
#define BUILTIN_NUMPY_H
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;


namespace builtin {
    template<typename tuple_t>
    constexpr auto tupleToArray(tuple_t&& tuple) {
        constexpr auto get_array = [](auto&&... x) { return std::array{std::forward<decltype(x)>(x)...}; };
        return std::apply(get_array, std::forward<tuple_t>(tuple));
    }

    struct view_info {
        bool selective;
        size_t offset;
        size_t end;
        size_t step;
    };

    template<typename T, size_t D>
    void dimensions_from_nested_list(const T& list, size_t pos, std::array<int64_t, D>& dimensions);
    template<typename T, size_t D>
    void dimensions_from_nested_list(const std::shared_ptr<std::vector<T>>& list, size_t pos, std::array<int64_t, D>& dimensions);
    template<typename T, size_t D>
    void dimensions_from_nested_list_helper(const T& list, size_t pos, std::array<int64_t, D>& dimensions) {
        if (dimensions[pos] >= 0) {
            if (dimensions[pos] != list.size()) {
                throw std::runtime_error("Inconsistent shape");
            }
        }
        dimensions[pos] = list.size();
        for (const auto& v : list) {
            if (pos + 1 < D) {
                dimensions_from_nested_list(v, pos + 1, dimensions);
            }
        }
    }

    template<typename T, size_t D>
    void dimensions_from_nested_list(const std::shared_ptr<std::vector<T>>& list, size_t pos, std::array<int64_t, D>& dimensions) {
        dimensions_from_nested_list_helper(*list, pos, dimensions);
    }

    template<typename T, size_t D>
    void dimensions_from_nested_list(const T& scalar, size_t pos, std::array<int64_t, D>& dimensions) {
    }
    template<size_t ND, size_t OD>
    std::tuple<size_t, std::array<int64_t, ND>, std::array<int64_t, ND>> computeOffsetAndStrides(
        std::vector<view_info> view_infos, std::array<int64_t, OD> o_strides, size_t o_offset) {
        size_t offset = 0;
        for (size_t i = 0; i < OD; i++) {
            offset += view_infos[i].offset * o_strides[i];
        }
        for (size_t i = 0; i < OD; i++) {
            auto& view_info = view_infos[i];
            auto step = view_info.step;
            o_strides[i] = o_strides[i] * step;
        }
        std::array<int64_t, ND> strides;
        std::array<int64_t, ND> shapes;
        size_t j = 0;
        for (size_t i = 0; i < OD; i++) {
            auto& view_info = view_infos[i];
            if (!view_info.selective) {
                auto start = view_info.offset;
                auto stop = view_info.end;
                auto step = view_info.step;
                shapes[j] = (stop - start + step - 1) / step;
                strides[j] = o_strides[i];
                j++;
            }
        }
        return std::make_tuple(offset + o_offset, shapes, strides);
    }

    template<size_t ND, size_t OD>
    std::tuple<bool, std::array<int64_t, ND>> try_reshape(std::array<int64_t, OD> olddims_,
                                                          std::array<int64_t, OD> oldstrides_,
                                                          std::array<int64_t, ND> newdims) {
        std::array<int64_t, ND> newstrides;
        int64_t last_stride = 1;
        int oi, oj, ni, nj, nk;

        /*
         * Remove axes with dimension 1 from the old array. They have no effect
         * but would need special cases since their strides do not matter.
         */
        std::vector<int64_t> olddims;
        std::vector<int64_t> oldstrides;
        for (oi = 0; oi < OD; oi++) {
            if (olddims_[oi] != 1) {
                olddims.push_back(olddims_[oi]);
                oldstrides.push_back(oldstrides_[oi]);
            }
        }
        int64_t oldnd = olddims.size();

        /* oi to oj and ni to nj give the axis ranges currently worked with */
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;
        while (ni < ND && oi < oldnd) {
            int64_t np = newdims[ni];
            int64_t op = olddims[oi];

            while (np != op) {
                if (np < op) {
                    /* Misses trailing 1s, these are handled later */
                    np *= newdims[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

            /* Check whether the original axes can be combined */
            for (int64_t ok = oi; ok < oj - 1; ok++) {
                /* C order */
                if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
                    /* not contiguous enough */
                    return {false, {}};
                }
            }

            /* Calculate new strides for all axes currently worked with */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk] * newdims[nk];
            }
            ni = nj++;
            oi = oj++;
        }

        /*
         * Set strides corresponding to trailing 1s of the new shape.
         */
        if (ni >= 1) {
            last_stride = newstrides[ni - 1];
        }
        for (nk = ni; nk < ND; nk++) {
            newstrides[nk] = last_stride;
        }

        return {true, newstrides};
    }

    template<class EleT>
    struct ndarray_data {
        EleT* data;

    public:
        ndarray_data(size_t num_elements) {
            data = new EleT[num_elements];
        }

        EleT* get_data() {
            return data;
        }

        ~ndarray_data() {
            delete[] data;
        }
    };

    template<class EleT, size_t D>
    class ndarray {
        std::array<int64_t, D> dimensions;
        std::array<int64_t, D> strides;
        std::shared_ptr<ndarray_data<EleT>> array_data;
        size_t offset;
        EleT* data;
        size_t num_elements;
        bool view;

    public:
        ndarray(std::array<int64_t, D> dimensions): dimensions(dimensions) {
            num_elements = 1;
            for (auto d: dimensions) {
                num_elements *= d;
            }
            strides[D - 1] = 1;
            for (size_t i = 1; i < D; i++) {
                strides[D - i - 1] = strides[D - i] * dimensions[D - i];
            }
            offset = 0;
            array_data = std::make_shared<ndarray_data<EleT>>(num_elements);
            data = array_data->get_data();
            view = false;
        }

        ndarray(std::shared_ptr<ndarray_data<EleT>> array_data, std::array<int64_t, D> dimensions,
                std::array<int64_t, D> strides, size_t offset,bool view=true): array_data(array_data), dimensions(dimensions),
                                                                strides(strides), offset(offset),view(view) {
            num_elements = 1;
            for (auto d: dimensions) {
                num_elements *= d;
            }
            data = array_data->get_data();
        }

        size_t size() {
            return num_elements;
        }

        std::shared_ptr<ndarray_data<EleT>> get_data() {
            return array_data;
        }

        std::array<int64_t, D>& shape() {
            return dimensions;
        }

        std::array<int64_t, D>& getStrides() {
            return strides;
        }

        size_t getOffset() {
            return offset;
        }

        inline int64_t getDimension(size_t index) {
            return dimensions[index];
        }

        EleT& byOffset(size_t index) {
            return data[index];
        }

        inline EleT& byIndices(std::array<int64_t, D> indices) {
            size_t index = offset;
            for (size_t i = 0; i < D; ++i) {
                index += strides[i] * indices[i];
            }
            return data[index];
        }

        py::object to_numpy() {
            if (view) {
                return materialize()->to_numpy();
            } else {
                auto result = py::array_t<EleT>(dimensions);
                std::memcpy(result.mutable_data(), data, num_elements * sizeof(EleT));
                return result;
            }
        }

        template<class OtherEleT>
        std::shared_ptr<ndarray<OtherEleT, D>> with_same_dims() {
            return std::make_shared<ndarray<OtherEleT, D>>(dimensions);
        }
        std::shared_ptr<ndarray<EleT,D>> materialize() {
            auto tempData_ = std::make_shared<ndarray_data<EleT>>(num_elements);
            auto tempData= tempData_->get_data();
            std::array<int64_t, D> natural_strides;
            natural_strides[D - 1] = 1;
            for (size_t i = 1; i < D; i++) {
                natural_strides[D - i - 1] = natural_strides[D - i] * dimensions[D - i];
            }
            for (size_t i = 0; i < num_elements; ++i) {
                size_t index = offset;
                auto idx = i;
                for (size_t j = 0; j < D; j++) {
                    index += strides[j] * (idx / natural_strides[j]);
                    idx = idx % natural_strides[j];
                }
                tempData[i] = data[index];
            }
            return std::make_shared<ndarray<EleT, D>>(tempData_, dimensions, natural_strides, 0,false);
        }

        template<size_t ND>
        std::shared_ptr<ndarray<EleT, ND>> reshape(std::array<int64_t, ND> newdims) {
            auto [success, newstrides] = try_reshape<ND,D>(dimensions, strides, newdims);
            if(success) {
                return std::make_shared<ndarray<EleT, ND>>(array_data,newdims, newstrides, offset,true);
            }else {
                return materialize()->reshape(newdims);
            }

        }
    };
}

#endif //BUILTIN_NUMPY_H
