from mnist.service import Service
import matplotlib.pyplot as plt


class Controller:
    def __init__(self):
        pass


    def run(self):
        service = Service()
        model = service.create_model()
        # i, predictions_array, true_label, img
        predictions =  model[0]
        test_labels = model[1]
        img = model[2]


        i = 5
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        service.plot_image(i, predictions, test_labels, img)
        plt.subplot(1, 2, 2)
        service.plot_value_array(i, predictions, test_labels)
        plt.show()