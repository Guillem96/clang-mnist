#include <stdio.h>

#include "tensor.h"


int main() {
    unsigned int shape[] = {1, 4, 1};
    unsigned int indexer[] = {0, 0, 0};

    tensor_t* m = tensor_arange(0, 4, 1, shape, 3);

    tensor_print(m);
    tensor_specs(m);

    tensor_t* indexed = tensor_index(m, indexer, 3);
    tensor_print(indexed);
    tensor_specs(indexed);

    return 0;
}
