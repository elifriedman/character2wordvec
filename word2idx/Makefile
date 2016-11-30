TF_INC=/opt/anaconda/lib/python2.7/site-packages/tensorflow/include

word_to_idx_op_kernel.so: word_to_idx.cc
	g++ -std=c++11 -shared $< -o $@ -fPIC -I $(TF_INC) -O2 -undefined dynamic_lookup

clean:
	rm *.so
