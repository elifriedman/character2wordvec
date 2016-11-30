#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <string>

using namespace tensorflow;

REGISTER_OP("WordToIdx")
.Attr("max_wordlen: int")
.Input("words: string")
.Output("word_idxs: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
});


class WordToIdxOp : public OpKernel {
public:

    explicit WordToIdxOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the index of the value to preserve
        OP_REQUIRES_OK(context,
                context->GetAttr("max_wordlen", &_max_wordlen));
        // Check that preserve_index is positive
        OP_REQUIRES(context, _max_wordlen > 0,
                errors::InvalidArgument("Need max_wordlen > 0, got ",
                _max_wordlen));
    }

    void Compute(OpKernelContext* context) { // override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<std::string>();
        
        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({input.size(), _max_wordlen}),
        &output_tensor));
        auto output = output_tensor->flat<int32>();

        const int N = input.size();
        for (int i = 0; i < N; i++) {
            std::string word = input(i);
            const int size = word.size();
            OP_REQUIRES(context, _max_wordlen >= size,
                                errors::InvalidArgument("word size greater than max_wordlen"));
            int cur_idx = i*_max_wordlen;
            int j = 0;
            for (; j < size; ++j) {
                // convert all useful characters to the range 0 .. 94
                output(cur_idx + j) =  (word[j] % 94); 
            }
            for (; j < _max_wordlen; ++j) {
                output(cur_idx + j) = -1;
            }
        }
    }
private:
    int _max_wordlen;
};

REGISTER_KERNEL_BUILDER(Name("WordToIdx").Device(DEVICE_CPU), WordToIdxOp);