# import tensorflow as tf




# tf.compat.v1.disable_eager_execution()
# x = tf.Variable(3, name="x")
# y = tf.Variable(4, name="y")

# f = x*x*y + y + 2

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
# 	x.initializer.run()
# 	y.initializer.run()
# 	result = f.eval()
# 	print(result)

l = [1,2,3]
print(l.index(2))