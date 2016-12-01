#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <string>

using namespace tensorflow;

REGISTER_OP("WordToIdx")
.Input("words: string")
.Input("max_wordlen: int32")
.Output("word_idxs: int32")
.SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::DimensionHandle max_wordlen;
    TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &max_wordlen));
    
    shape_inference::ShapeHandle out;
    TF_RETURN_IF_ERROR(c->Concatenate(c->input(0), c->Vector(max_wordlen), &out));
    c->set_output(0, out);
    return Status::OK();
});


class WordToIdxOp : public OpKernel {
public:

    explicit WordToIdxOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context)  { // override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<std::string>();

        const Tensor& max_wordlen = context->input(1);
        auto tmp = max_wordlen.flat<int>();
        OP_REQUIRES(context, tmp.size() == 1,
                    errors::InvalidArgument("Need to provide one input for max_wordlen: ",tmp.size()));
        int _max_wordlen = tmp(0);
        OP_REQUIRES(context, _max_wordlen > 0,
                    errors::InvalidArgument("_max_wordlen needs to be greater than 0: ",_max_wordlen));
        
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