/*
 * model.h
 *
 * Created:  11/20/2016
 * Author:  Eli Friedman (friedm3@cooper.edu)
 *
 */
#ifndef MODEL_H
#define MODEL_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include <iostream>

template <class Builder>
class BiRNNModel {
    public:
        explicit BiRNNModel(dynet::Model& model,
                            int num_layers,
                            unsigned input_dim,
                            unsigned hidden_dim,
                            unsigned output_dim,
                            int num_chars) :
            l2rbuilder(num_layers, input_dim, hidden_dim, &model),
            r2lbuilder(num_layers, input_dim, hidden_dim, &model)
        {
            p_embed = model.add_lookup_parameters(num_chars, {input_dim}); // input "embedding" vectors
            p_leftHidden2output = model.add_parameters({output_dim, hidden_dim});  // from hidden -> output
            p_rightHidden2output = model.add_parameters({output_dim, hidden_dim}); // from hidden -> output
            p_outputBias = model.add_parameters({output_dim}); // output bias

            p_output = model.add_parameters({1, 2*output_dim});
            p_bias = model.add_parameters({1});
        }

        dynet::LookupParameter p_embed;
        dynet::Parameter p_leftHidden2output;
        dynet::Parameter p_rightHidden2output;
        dynet::Parameter p_outputBias;

        dynet::Parameter p_output;
        dynet::Parameter p_bias;

        Builder l2rbuilder;
        Builder r2lbuilder;

        Expression build(const std::vector<int>& input, dynet::ComputationGraph& cg);
        Expression getNCEModelOutput(const std::vector<int>& word,
                                     const std::vector<int>& context,
                                     dynet::ComputationGraph& cg);

        // reset RNN builder for new graph
        void new_graph(dynet::ComputationGraph& cg)
        {
            l2rbuilder.new_graph(cg);  
            r2lbuilder.new_graph(cg); 
        }

};

template <class Builder>
Expression BiRNNModel<Builder>::build(const std::vector<int>& input, dynet::ComputationGraph& cg)
{
    const unsigned len = input.size();
    l2rbuilder.start_new_sequence();
    r2lbuilder.start_new_sequence();

    Expression i_leftHidden2output  = parameter(cg, p_leftHidden2output);
    Expression i_rightHidden2output = parameter(cg, p_rightHidden2output);
    Expression i_outputBias         = parameter(cg, p_outputBias);

    for(unsigned t = 0; t < len; ++t){
        Expression x_forward = lookup(cg, p_embed, input[t]);
        Expression x_reverse = lookup(cg, p_embed, input[len - t - 1]);
        // if (!eval) { x_forward[t] = noise(x_forward[t], 0.1); }
        // if (!eval) { x_reverse[t] = noise(x_reverse[t], 0.1); }
        l2rbuilder.add_input(x_forward);
        r2lbuilder.add_input(x_reverse);
    }
    Expression l2routput = l2rbuilder.back(); // get last output of l2r rnn
    Expression r2loutput = r2lbuilder.back(); // get last output of r2l rnn

    // output = tanh( i_outputBias + i_leftHidden2output*l2routput + i_rightHidden2output*r2loutput)
    Expression output = tanh(affine_transform({i_outputBias,
                                               i_leftHidden2output, l2routput,
                                               i_rightHidden2output, r2loutput}));
    return output;
}

template <class Builder>
Expression BiRNNModel<Builder>::getNCEModelOutput(const std::vector<int>& word,
                                                  const std::vector<int>& context,
                                                  dynet::ComputationGraph& cg)
{
    Expression word_lstm = build(word, cg);
    Expression context_lstm = build(context, cg);
    Expression concat = concatenate({word_lstm, context_lstm});
    Expression i_output = parameter(cg, p_output);
    Expression i_bias = parameter(cg, p_bias);

    Expression tform = affine_transform({i_bias, i_output, concat});
    Expression output = rectify(tform);

    return output;
}


#endif // MODEL_H
