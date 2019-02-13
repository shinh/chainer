#pragma once

#include <cstdint>

#include "chainerx/constant.h"
#include "chainerx/error.h"
#include "chainerx/slice.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

enum class ArrayIndexTag {
    kSingleElement = 1,
    kSlice,
    kNewAxis,
};

class NewAxis {};

class ArrayIndex {
public:
    ArrayIndex(int64_t index) : tag_{ArrayIndexTag::kSingleElement}, index_{index} {}  // NOLINT(runtime/explicit)
    ArrayIndex(Slice slice) : tag_{ArrayIndexTag::kSlice}, slice_{slice} {}  // NOLINT(runtime/explicit)
    ArrayIndex(NewAxis) : tag_{ArrayIndexTag::kNewAxis} {}  // NOLINT(runtime/explicit)
    ArrayIndex() : tag_{ArrayIndexTag::kSlice} {}

    ArrayIndexTag tag() const { return tag_; }

    int64_t index() const {
        if (tag_ != ArrayIndexTag::kSingleElement) {
            throw ChainerxError{"Array index is not a single element."};
        }
        return index_;
    }

    Slice slice() const {
        if (tag_ != ArrayIndexTag::kSlice) {
            throw ChainerxError{"Array index is not a slice."};
        }
        return slice_;
    }

private:
    ArrayIndexTag tag_;
    int64_t index_{};
    Slice slice_;
};

// Up to `kMaxNdim * 2` indices since there can be `kMaxNdim`
// of indices with `kNewAxis` type.
typedef StackVector<ArrayIndex, kMaxNdim * 2> ArrayIndices;

}  // namespace chainerx
