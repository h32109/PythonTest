from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np




class Service:
    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] #모델한테 정답을 알려준다.


    def create_model(self) -> []:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


        train_images = train_images / 255.0
        test_images = test_images / 255.0


        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[train_labels[i]])
        # plt.show()


        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        """
        relu ( Recitified Linear Unit 정류한 선형 유닛)
        미분 가능한 0과 1사이의 값을 갖도록 하는 알고리즘
        softmax
        nn (neural network )의 최상위층에서 사용되며 classfication 을 위한 function
        결과를 확률값으로 해석하기 위한 알고리즘
        """


        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']) # 여러개의 층을 컴파일해서 하나의 모델로 만들기
        # learning 공부
        model.fit(train_images, train_labels, epochs=5) # 모델과 답을 최적화 시킨다. epochs는 학습 5회 6만개 학습x5
        # test 시험
        test_loss, test_acc = model.evaluate(test_images, test_labels) #평가


        print('\n 테스트 정확도: ', test_acc)


        # prediction 수능
        predictions = model.predict(test_images)
        print(predictions[3])
        # 10개 클래스에 대한 예측을 그래프화
        arr = [predictions,test_labels,test_images]
        return arr


    def plot_image(self, i, predictions_array, true_label, img)->[]:
        print(' === plot_image 로 진입 ===')
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])


        plt.imshow(img, cmap=plt.cm.binary)
        # plt.show()
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'


        plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             self.class_names[true_label]),
                   color=color)
    @staticmethod
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)


        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')