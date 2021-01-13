#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include "tensor.h"
#include "tensor_pool.h"
#include "mnist.h"
#include "plot.h"
#include "nn.h"

#define TRAIN_IMAGES "data/train-images-idx3-ubyte"
#define TRAIN_LABELS "data/train-labels-idx1-ubyte"
#define TEST_IMAGES "data/t10k-images-idx3-ubyte"
#define TEST_LABELS "data/t10k-labels-idx1-ubyte"


typedef struct 
{
    tensor_t* dW1;
    tensor_t* db1;

    tensor_t* dW2;
    tensor_t* db2;

    float loss;
    tensor_t* predictions;
} train_res_t;

static tensor_t* update_parameter(const tensor_t* param, const tensor_t* grad, float lr);

static tensor_t* forward(const tensor_t* x,
        const tensor_t* W1, const tensor_t* b1,
        const tensor_t* W2, const tensor_t* b2);

static train_res_t forward_backward(const tensor_t* x, const tensor_t* y,
        const tensor_t* W1, const tensor_t* b1,
        const tensor_t* W2, const tensor_t* b2);

static tensor_t* layer_init(uint32_t* shape, uint32_t n_dims);

int main(int argc, char** argv) 
{
    srand(time(NULL));

    mnist_t* ds, *test_ds;
    mnist_example_t* batch;
    tensor_pool_t* train_pool = tensor_pool_init();

    /* Training Hyperparams */
    int batch_size = 256;
    float lr = 0.001;

    /* Monitoring variables */
    train_res_t train_res;
    float acc = 0, loss = 0, test_acc;

    /* NN architecture parameters */
    uint32_t l1_shape[] = {28 * 28, 128};
    uint32_t l2_shape[] = {128, 10};

    /* Create the Neural Network 
     * An NN is no more than a set of matrices
     */
    tensor_t* W1 = layer_init(l1_shape, 2);
    tensor_t* b1 = layer_init(&l1_shape[1], 1);
    tensor_t* W2 = layer_init(l2_shape, 2);
    tensor_t* b2 = layer_init(&l2_shape[1], 1);
    tensor_t* test_preds;

    /* Load the train and test data */
    ds = mnist_read(TRAIN_IMAGES, TRAIN_LABELS);
    test_ds = mnist_read(TEST_IMAGES, TEST_LABELS);

    for (int step = 0; step < 250; step++)
    {
        /* Randomly sample a batch  
         * we set the last parameter to 1 indicating that we want the flattened
         * version of the batch (shape of [batch_size, 28 * 28]) 
         */
        batch = mnist_batch(ds, batch_size, 1);
        tensor_pool_add(train_pool, batch->image);
        tensor_pool_add(train_pool, batch->label);

        /* Run the forward pass and also compute the gradients */
        train_res = forward_backward(batch->image, batch->label, W1, b1, W2, b2);
        acc += nn_accuracy_score(batch->label, train_res.predictions);
        loss += train_res.loss;

        /* Update second layer parameters */
        tensor_pool_add(train_pool, train_res.dW2);
        tensor_pool_add(train_pool, W2);
        W2 = update_parameter(W2, train_res.dW2, lr);

        tensor_pool_add(train_pool, train_res.db2);
        tensor_pool_add(train_pool, b2);
        b2 = update_parameter(b2, train_res.db2, lr);

        /* Update first layer parameters */
        tensor_pool_add(train_pool, train_res.dW1);
        tensor_pool_add(train_pool, W1);
        W1 = update_parameter(W1, train_res.dW1, lr);

        tensor_pool_add(train_pool, train_res.db1);
        tensor_pool_add(train_pool, b1);
        b1 = update_parameter(b1, train_res.db1, lr);
        tensor_pool_empty(train_pool);

        if ((step + 1) % 20 == 0)
            printf("[Step %d] loss: %.5f  accuracy: %.5f\n", step, train_res.loss / step, acc / step);
    }
    tensor_pool_clean(train_pool);

    printf("Running test evaluation... ");
    batch = mnist_as_tensor(test_ds, 1);
    test_preds = forward(batch->image, W1, b1, W2, b2);
    test_acc = nn_accuracy_score(batch->label, test_preds);
    printf("Test accuracy: %.2f\n", test_acc * 100);

    mnist_clean(ds);
    mnist_clean(test_ds);

    tensor_clean(batch->image);
    tensor_clean(batch->label);
    tensor_clean(test_preds);

    tensor_clean(W1);
    tensor_clean(W2);
    tensor_clean(b1);
    tensor_clean(b2);

    return 0;
}

tensor_t* update_parameter(const tensor_t* param, const tensor_t* grad, float lr)
{
    tensor_pool_t* pool = tensor_pool_init();
    tensor_t* res;

    grad = tensor_mul_scalar(grad, lr);
    tensor_pool_add(pool, grad);

    res = tensor_sub(param, grad);

    tensor_pool_clean(pool);

    return res;
}

tensor_t* forward(const tensor_t* x,
        const tensor_t* W1, const tensor_t* b1,
        const tensor_t* W2, const tensor_t* b2)
{
    tensor_t* t1 = tensor_mm(x, W1);
    tensor_t* z1 = tensor_add(t1, b1);
    tensor_t* a1 = nn_relu(z1);

    tensor_t* t2 = tensor_mm(a1, W2);
    tensor_t* z2 = tensor_add(t2, b2);
    tensor_t* a2 = nn_softmax(z2, 1);
    tensor_t* preds = tensor_argmax(a2, 1);

    tensor_clean(t1);
    tensor_clean(z1);
    tensor_clean(a1);
    tensor_clean(t2);
    tensor_clean(z2);
    tensor_clean(a2);
    return preds;
}

static train_res_t forward_backward(const tensor_t* x, const tensor_t* y,
        const tensor_t* W1, const tensor_t* b1,
        const tensor_t* W2, const tensor_t* b2)
{
    tensor_pool_t* pool = tensor_pool_init();

    float piy;
    train_res_t res;

    tensor_t* dz2;
    tensor_t* da1;
    tensor_t* dz1;

    tensor_t* t1 = tensor_mm(x, W1);
    tensor_pool_add(pool, t1);

    tensor_t* z1 = tensor_add(t1, b1);
    tensor_pool_add(pool, z1);

    tensor_t* a1 = nn_relu(z1);
    tensor_pool_add(pool, a1);

    tensor_t* t2 = tensor_mm(a1, W2);
    tensor_pool_add(pool, t2);

    tensor_t* z2 = tensor_add(t2, b2);
    tensor_pool_add(pool, z2);

    tensor_t* a2 = nn_softmax(z2, 1);
    tensor_pool_add(pool, a2);

    res.predictions = tensor_argmax(a2, 1);
    res.loss = nn_sparse_ce_loss(y, a2);

    dz2 = tensor_copy(a2);
    for (int i = 0; i < tensor_numel(y); i++)
    {
        piy = a2->values[i * a2->shape[1] + (int)y->values[i]];
        dz2->values[i * dz2->shape[1] + (int)y->values[i]] = piy - 1 ;
    }
    tensor_pool_add(pool, dz2);

    dz2 = tensor_div_scalar(dz2, x->shape[0]);
    tensor_pool_add(pool, dz2);

    res.dW2 = tensor_mm(tensor_T(a1), dz2);
    res.db2 = tensor_reduce_sum(dz2, 0);

    da1 = tensor_mm(dz2, tensor_T(W2));
    tensor_pool_add(pool, da1);

    dz1 = tensor_zeros(z1->shape, 2);
    tensor_pool_add(pool, dz1);

    dz1 = tensor_gte(z1, dz1);
    tensor_pool_add(pool, dz1);

    dz1 = tensor_mul(dz1, da1);
    tensor_pool_add(pool, dz1);

    res.dW1 = tensor_mm(tensor_T(x), dz1);
    res.db1 = tensor_reduce_sum(dz1, 0);

    tensor_pool_clean(pool);

    return res;
}

static tensor_t* layer_init(uint32_t* shape, uint32_t n_dims)
{
    float prod = 1;
    tensor_t* gc, *res;

    for (int i = 0; i < n_dims; i++)
        prod *= shape[i];

    gc = tensor_uniform(-1, 1, shape, n_dims);
    res = tensor_div_scalar(gc, sqrt(prod));
    tensor_clean(gc);
    return res;
}


