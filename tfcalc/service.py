import tensorflow as tf
from tfcalc.model import Model
class Service:
    def __init__(self):
        self._this = Model()

    @tf.function
    def plus(self, this):
        return tf.add(this.num1, this.num2)

    @tf.function
    def minus(self, this):
        return tf.subtract(this.num1, this.num2)

    @tf.function
    def multiple(self, this):
        return tf.multiply(this.num1, this.num2)

    @tf.function
    def devide(self, this):
        return tf.divide(this.num1, this.num2)