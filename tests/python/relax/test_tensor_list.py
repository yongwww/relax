import tensorflow as tf
import numpy as np


def test_tensor_array_stack():
    def run(dtype_str, infer_shape):

        with tf.Graph().as_default():
            dtype = tf.float32
            t = tf.constant(np.array([[1.0], [2.0], [3.0]]).astype(dtype_str))
            scatter_indices = tf.constant([2, 1, 0])
            ta1 = tf.TensorArray(dtype=dtype, size=3, infer_shape=infer_shape)
            ta2 = ta1.scatter(scatter_indices, t)
            t1 = ta2.stack()

            print(" yongwww: \n", t1)
            # print(t1.numpy())
            res = tf.make_ndarray(t1)
            print(res)

            g = tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)
            # print(g)
            print("done")

    for dtype in ["float32"]:
        run(dtype, True)


def test_tensor_array_write():
    import tensorflow as tf

    ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, infer_shape=False)
    ta = ta.write(0, 1)
    ta = ta.write(1, (1, 2))
    ta = ta.write(2, ((1, 2, 3)))
    tensors = [ta.read(i) for i in range(3)]
    print(tensors)


def test_tensorlist_stack_model():
    def tensorlist_stack_model(input_shape):
        class TensorArrayStackLayer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()

            def call(self, inputs):
                inputs = tf.squeeze(inputs)
                outputs = tf.TensorArray(
                    tf.float32,
                    size=inputs.shape[0],
                    infer_shape=False,
                    element_shape=inputs.shape[1:],
                )
                outputs = outputs.unstack(inputs)

                return outputs.stack()

        input_shape = (3, 32)
        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=input_shape, batch_size=1), TensorArrayStackLayer()]
        )
        return model

    run_sequential_model(tensorlist_stack_model, input_shape=(3, 32))


if __name__ == "__main__":
    test_tensor_array_stack()
