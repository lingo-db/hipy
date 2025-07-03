#ifndef BUILTIN_ARROW_H
#define BUILTIN_ARROW_H
#include "builtin_commons.h"
#include <arrow/table.h>
#include <arrow/builder.h>
#include <arrow/python/pyarrow.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include<pybind11/embed.h>
namespace py = pybind11;

namespace builtin::tabular {
    class Column {
        std::shared_ptr<arrow::ChunkedArray> data;

    public:
        explicit Column(std::shared_ptr<arrow::ChunkedArray> data): data(data) {
        }

        py::object to_python() {
            return py::cast<py::object>(arrow::py::wrap_chunked_array(data));
        }

        std::shared_ptr<arrow::ChunkedArray> getData() {
            return data;
        }

        template<class AccessorT, class Fn>
        void iterate(Fn fn) {
            for (const auto& chunk: data->chunks()) {
                auto array = chunk;
                AccessorT accessor(array);
                for (size_t i = 0; i < array->length(); i++) {
                    fn(accessor.access(i));
                }
            }
        }

        template<class Accessor1T, class Accessor2T, class Fn>
        void iterateZipped(std::shared_ptr<Column>& other, Fn fn) {
            auto chunksLeft = data->chunks();
            auto chunksRight = other->getData()->chunks();
            size_t leftChunkIndex = 0;
            size_t rightChunkIndex = 0;
            size_t leftIndex = 0;
            size_t rightIndex = 0;
            while (leftChunkIndex < chunksLeft.size() && rightChunkIndex < chunksRight.size()) {
                auto leftArray = std::static_pointer_cast<arrow::Array>(chunksLeft[leftChunkIndex]);
                auto rightArray = std::static_pointer_cast<arrow::Array>(chunksRight[rightChunkIndex]);
                Accessor1T leftAccessor(leftArray);
                Accessor2T rightAccessor(rightArray);
                while (leftIndex < leftArray->length() && rightIndex < rightArray->length()) {
                    fn(leftAccessor.access(leftIndex), rightAccessor.access(rightIndex));
                    leftIndex++;
                    rightIndex++;
                }
                if (leftIndex == leftArray->length()) {
                    leftIndex = 0;
                    leftChunkIndex++;
                }
                if (rightIndex == rightArray->length()) {
                    rightIndex = 0;
                    rightChunkIndex++;
                }
            }
        }

        std::shared_ptr<Column> unique() {
            return std::make_shared<Column>(
                arrow::ChunkedArray::Make({arrow::compute::Unique(data).ValueOrDie()}).ValueOrDie());
        }

        std::shared_ptr<Column> isIn(std::shared_ptr<Column> other) {
            auto boolv = arrow::compute::IsIn(data, other->getData()).ValueOrDie().chunked_array();
            return std::make_shared<Column>(boolv);
        }

        int64_t size() {
            return data->length();
        }

        template<class AccessorT>
        AccessorT::element_type byIndex(int64_t idx) {
            size_t chunk_offset = 0;
            for (const auto& chunk: data->chunks()) {
                auto array = std::static_pointer_cast<arrow::Array>(chunk);
                if (idx < chunk_offset + array->length()) {
                    return AccessorT(array).access(idx - chunk_offset);
                }
                chunk_offset += array->length();
            }
            throw std::runtime_error("Index out of bounds");
        }
    };

    template<class ArrType, class ElType>
    class GenericColumnAccessor {
        std::shared_ptr<ArrType> array;

    public:
        using array_type = ArrType;
        using element_type = ElType;


        explicit GenericColumnAccessor(std::shared_ptr<arrow::Array> array): array(
            std::static_pointer_cast<ArrType>(std::move(array))) {
        }

        ElType access(size_t pos) {
            return array->Value(pos);
        }
    };

    using BoolColumnAccessor = GenericColumnAccessor<arrow::BooleanArray, bool>;
    using Int8ColumnAccessor = GenericColumnAccessor<arrow::Int8Array, int8_t>;
    using Int16ColumnAccessor = GenericColumnAccessor<arrow::Int16Array, int16_t>;
    using Int32ColumnAccessor = GenericColumnAccessor<arrow::Int32Array, int32_t>;
    using Int64ColumnAccessor = GenericColumnAccessor<arrow::Int64Array, int64_t>;
    using Float32ColumnAccessor = GenericColumnAccessor<arrow::FloatArray, float>;
    using Float64ColumnAccessor = GenericColumnAccessor<arrow::DoubleArray, double>;

    class StrColumnAccessor {
        std::shared_ptr<arrow::StringArray> array;

    public:
        using array_type = arrow::StringArray;
        using element_type = std::string;

        explicit StrColumnAccessor(std::shared_ptr<arrow::Array> array): array(
            std::static_pointer_cast<arrow::StringArray>(std::move(array))) {
        }

        std::string access(size_t pos) {
            return array->GetString(pos);
        }
    };

    template<class ElementAccessorT>
    class ListColumnAccessor {
        std::shared_ptr<arrow::ListArray> array;

    public:
        using array_type = arrow::ListArray;
        using element_type = List<typename ElementAccessorT::element_type>;

        explicit ListColumnAccessor(std::shared_ptr<arrow::Array> array): array(
            std::static_pointer_cast<arrow::ListArray>(std::move(array))) {
        }

        std::shared_ptr<std::vector<typename ElementAccessorT::element_type>> access(size_t pos) {
            auto subarray = std::static_pointer_cast<typename
                ElementAccessorT::array_type>(array->value_slice(pos));
            ElementAccessorT subAccessor(subarray);
            std::vector<typename ElementAccessorT::element_type> result;
            for (size_t i = 0; i < subarray->length(); i++) {
                result.push_back(subAccessor.access(i));
            }
            return std::make_shared<std::vector<typename ElementAccessorT::element_type>>(result);
        }
    };
    template<class FB>
    void reserveData(FB& builder, size_t size) {
    }
    void reseveData(arrow::StringBuilder& builder, size_t size) {
        builder.ReserveData(size*4);
    }
    template<class E, class B>
    class ColumnBuilder {
        B builder;
        size_t numValues;
        std::vector<std::shared_ptr<arrow::Array>> additionalArrays;
    public:
        using builder_type = B;
        using element_type = E;
        ColumnBuilder() :numValues(0){
            auto reserveOk=builder.Reserve(20000).ok();
            assert(reserveOk);
            reserveData(builder, 20000);

        }

        void append(E element) {
            numValues++;
            if(numValues>20000) {
                auto array = builder.Finish().ValueOrDie();
                additionalArrays.push_back(array);
                auto reserveOk=builder.Reserve(20000).ok();
                assert(reserveOk);
                numValues=0;
            }
            builder.Append(element);
        }

        std::shared_ptr<Column> build() {
            std::shared_ptr<arrow::Array> arr = builder.Finish().ValueOrDie();
            additionalArrays.push_back(arr);
            return std::make_shared<Column>(std::make_shared<arrow::ChunkedArray>(additionalArrays));
        }
    };

    using Float64ColumnBuilder = ColumnBuilder<double, arrow::DoubleBuilder>;
    using Float32ColumnBuilder = ColumnBuilder<float, arrow::FloatBuilder>;
    using Int32ColumnBuilder = ColumnBuilder<int32_t, arrow::Int32Builder>;
    using Int16ColumnBuilder = ColumnBuilder<int16_t, arrow::Int16Builder>;
    using Int8ColumnBuilder = ColumnBuilder<int8_t, arrow::Int8Builder>;
    using Int64ColumnBuilder = ColumnBuilder<int64_t, arrow::Int64Builder>;
    using StrColumnBuilder = ColumnBuilder<std::string, arrow::StringBuilder>;
    using BoolColumnBuilder = ColumnBuilder<bool, arrow::BooleanBuilder>;

    template<class CB>
    class ListColumnBuilder {
        arrow::ListBuilder builder;

    public:
        ListColumnBuilder(): builder(
            arrow::ListBuilder(arrow::default_memory_pool(), std::make_shared<typename CB::builder_type>())) {
        }

        void append(std::shared_ptr<std::vector<typename CB::element_type>> element) {
            builder.Append().ok();
            for (auto e: *element) {
                reinterpret_cast<typename CB::builder_type *>(builder.value_builder())->Append(e);
            }
        }

        std::shared_ptr<Column> build() {
            std::shared_ptr<arrow::Array> arr = builder.Finish().ValueOrDie();
            return std::make_shared<Column>(std::make_shared<arrow::ChunkedArray>(arr));
        }
    };

    class Table {
        std::shared_ptr<arrow::Table> data;

    public:
        Table(std::shared_ptr<arrow::Table> data): data(data) {
        }

        static std::shared_ptr<Table> from_columns(
            std::vector<std::pair<std::string, std::shared_ptr<Column>>> columns) {
            std::vector<std::shared_ptr<arrow::Field>> fields;
            std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
            for (auto column: columns) {
                fields.push_back(arrow::field(column.first, column.second->getData()->type()));
                arrays.push_back(column.second->getData());
            }
            auto schema = std::make_shared<arrow::Schema>(fields);
            auto table = arrow::Table::Make(schema, arrays);
            return std::make_shared<Table>(table);
        }

        py::object to_python() {
            return py::cast<py::object>(arrow::py::wrap_table(data));
        }

        std::shared_ptr<Table> sort(List<std::string> names, List<bool> ascending) {
            arrow::compute::SortOptions options;
            for (auto i = 0ull; i < names->size(); i++) {
                auto columnName = (*names)[i];
                auto order = (*ascending)[i];
                options.sort_keys.push_back(arrow::compute::SortKey(
                    columnName, order ? arrow::compute::SortOrder::Ascending : arrow::compute::SortOrder::Descending));
            }
            auto indices = arrow::compute::SortIndices(data, options).ValueOrDie();
            auto sorted = arrow::compute::Take(data, indices).ValueOrDie().table();
            return std::make_shared<Table>(sorted);
        }

        template<class Fn>
        void iterateBatches(Fn fn) {
            arrow::TableBatchReader reader(*data);
            std::shared_ptr<arrow::RecordBatch> batch;
            while (reader.ReadNext(&batch).ok() && batch) {
                fn(batch);
            }
        }

        std::shared_ptr<Table> slice(int64_t start, int64_t end) {
            return std::make_shared<Table>(data->Slice(start, end - start));
        }

        int64_t size() {
            return data->num_rows();
        }

        std::shared_ptr<Column> getColumn(std::string name) {
            return std::make_shared<Column>(data->GetColumnByName(name));
        }

        std::shared_ptr<Table> selectColumns(std::vector<std::string> columns) {
            std::vector<int> indices;
            for (auto& name: columns) {
                indices.push_back(data->schema()->GetFieldIndex(name));
            }
            return std::make_shared<Table>(data->SelectColumns(indices).ValueOrDie());
        }
        std::shared_ptr<Table> selectColumns(List<std::string> columns) {
            std::vector<int> indices;
            for (auto& name: *columns) {
                indices.push_back(data->schema()->GetFieldIndex(name));
            }
            return std::make_shared<Table>(data->SelectColumns(indices).ValueOrDie());
        }

        std::shared_ptr<Table> setColumn(std::string name, std::shared_ptr<Column> column) {
            auto fields = data->schema()->fields();
            auto arrays = data->columns();
            auto idx = data->schema()->GetFieldIndex(name);
            if (idx == -1) {
                arrays.push_back(column->getData());
                fields.push_back(arrow::field(name, column->getData()->type()));
            } else {
                arrays[idx] = column->getData();
                fields[idx] = arrow::field(name, column->getData()->type());
            }
            auto schema = std::make_shared<arrow::Schema>(fields);
            return std::make_shared<Table>(arrow::Table::Make(schema, arrays));
        }

        std::shared_ptr<Table> filterByColumn(std::shared_ptr<Column> mask) {
            auto filtered = arrow::compute::Filter(data, mask->getData()).ValueOrDie().table();
            return std::make_shared<Table>(filtered);
        }
    };
	std::shared_ptr<Table> load(std::string path) {
        auto inputFile = arrow::io::ReadableFile::Open(path).ValueOrDie();
        auto batchReader = arrow::ipc::RecordBatchFileReader::Open(inputFile).ValueOrDie();
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (int i = 0; i < batchReader->num_record_batches(); i++) {
           batches.push_back(batchReader->ReadRecordBatch(i).ValueOrDie());
        }
        return std::make_shared<Table>(arrow::Table::FromRecordBatches(batchReader->schema(), batches).ValueOrDie());
    }
}

#endif //BUILTIN_ARROW_H
