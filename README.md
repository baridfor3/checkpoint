# checkpoint
This is the trained model for our submitted paper.

Due to github large-file restrictions, the model is submitted via "git-lfs" so please us "git clone" to get all the files instead of downloading.


After cloning this project, you can move the checkpoint to lazy_transformer folder and then use debug.py for evaluation.


Reminder:

1. The presented method is in   !!!  lazy_transformer/lazyTransition.py !!!

2. The CKA script is in !!! UNIVERSAL/utils/cka.py !!!

3. All source code is submitted.

Hint:

1.For LT blocks (lazy_transformer/lazyTransition.py), we have:
  
   def call(self, x, *args, **kwargs):
        training = kwargs["training"]
        #self.layer is a UT block
	#  y is the UT(h{t-1})
	#  x is the h{t-1}
	# self.halting_pro= [batch_size, seq_length, sigma] is the sigma
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

3. We provide our debug.py. Users can load our trained model for evaluation. See comments in debug.py.


4. If you want to retrain the model, please delete the checkpoint or move outside the folder and run main.py.

5. Meanwhile, we provide UNIVERSAL/util/vis/heatmap.py for visualization. See comments in debug.py for generating visualization data.
