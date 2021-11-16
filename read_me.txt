NOTE THAT: This is a preview version.

######################################################################################
There are two projects in this folder.

UNIVERSAL is our basic project handing general layers like self-attention.

lazy_transformer is the paper code.

We release our vocabulary in this folder and our trained model (https://github.com/baridfor3/checkpoint.git) for(En -> De).



Hints:
1. The BPE vocabulary are in UNIVERSAL/vocabulary  (En -> De)

2. UNIVERSAL/tokenizer provides tools for generating BPE

3. The presented method is in   !!!  lazy_transformer/lazyTransition.py !!!

4. The CKA script is in !!! UNIVERSAL/utils/cka.py !!!

5. Use git to get the trained model. Then, just mv the pulled folder to lazy_transformer. Now, you can use debug.py to load our model. See comments in debug.py.
Usage:
1. Download dataset from WMT.
2. Run UNIVERSAL/tokenizer/py_tokenlizer.py for all the dataset.
3. Run UNIVERSAL/tokenizer/bpe.py for BPE codes (we have submitted our {En, De} in UNIVERSAL/vocabulary for NMT )
4. Set up the dataset path in lazy_transformer/initialization.py/offline.
6. Set up all the configurations in lazy_transformer/configuration.py


Implementation logic (also see comments in  lazy_transformer/lazyTransition.py  and lazy_transformer/lt.py ):

1. For LT blocks, we have:
  
   def call(self, x, *args, **kwargs):
        training = kwargs["training"]
        #self.layer is a UT block
	#  y is the UT(h{t-1})
	#  x is the h{t-1}
	# self.halting_pro is the sigma
	# Refer to Eq.6 in the paper
        y = self.layer(x, *args, **kwargs)
        self.halting_pro = tf.stop_gradient(cka.feature_space_linear_cka(x, y))
        if training:
            y = tf.nn.dropout(y, self.dropout)
        y = self.output_norm(x + (1 - self.halting_pro) * y)
        return y

2. cka (UNIVERSAL/utils/cka.py) is based on officially code:  https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
   ONE CHANGE: we extend the last dimension so we set the feature space to the channel.

    dot_product_similarity = (
        tf.norm(tf.matmul(tf.expand_dims(features_x, -1), tf.expand_dims(features_y, -1), transpose_a=True), axis=-1)
        ** 2
    )
    # normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_x = tf.norm(
        tf.matmul(tf.expand_dims(features_x, -1), tf.expand_dims(features_x, -1), transpose_a=True), axis=-1
    )

    # normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    normalization_y = tf.norm(
        tf.matmul(tf.expand_dims(features_y, -1), tf.expand_dims(features_y, -1), transpose_a=True), axis=-1
    )

3. We provide our debug.py. Users can load our trained model for evaluation.

4. If you want to retrain the model, please delete the checkpoint or move outside the folder and run main.py.