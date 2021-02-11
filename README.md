# Audio_denosing

В данной работе представлено решение задачи зашумленных акустических данных при помощи шумоподавляющего автокодировщика.

**Шаг 1.** Был взят датасет RAVDESS (база данных с набором речевых данных канадского университета Райерсон). (The Ryerson Audio-Visual Database of Emotional Speech and Song – RAVDESS. Она содержит 7356 файлов (общий размер: 24,8 ГБ). База данных сбалансирована по полу и включает в себя голоса 24 профессиональных актеров (12 женщин, 12 мужчин), озвучивающих два лексически подобранных высказывания с нейтральным североамериканским акцентом. Частота дискретизации каждого файла составляла 48 кГц.
(Выбор пал на него потому как было ранее я использовал его в своей бакалаврской работе, ничего не нужно было скачивать). Для тренировки использовались 200 аудиозаписей, 30 для валидации и 20 для теста.

**Шаг 2.** Была написана и зафиксирована процедуру зашумления данных. К предварительно отнормированным исходным сигналам добавлялись два вида шума - аддитивный (белый гауссовский) и мультипликативный (умножение сигнала на случайное число). 

Создание датасета, пригодного для работы с ним в PyTorch'e реализована в Audio_dataset.py

**Шаг 3.** Реализован шумоподавляющий автокодировщик для denoising audio с помощью фреймворка PyTorch языка Python. В качестве признаков использованы нормированный вэйформы сигналов (на вход подавались зашумленные данные, на выход - очищенные). Сама реализация сети описана в Autoencoder.py.

**Шаг 4.** Обучение модели проходило с лоссом MSE, в качестве оптимизатора среди прочих наилучшие результаты достиг Adam. Само обучение проходило на GPU, которую предоставляет компания Google в своем бесплатном облачном сервисе на основе Jupyter Notebook - Google Colab. Само обучение длилось в течение 10 эпох. Функция для тренировки сети описана в train_function.py, а сам непосредственно само обучение - в train.py.

**Шаг 5.** Тестирование системы описано в test.py. Ошибка на тренировочном на тесте составила 0.0506. 

## ВЫВОД:
Шумоподавляющий автокодировщик пригоден для задач audio denosing, разработанная система хорошо подавляет шумы, что видно на изображениях вэйформ ниже:

![12](https://user-images.githubusercontent.com/60327928/107619179-d2417780-6c63-11eb-9243-544e6b5922e6.png)

                                                              Fig.1 - Result of denoising signal
