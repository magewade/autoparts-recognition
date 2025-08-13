from dataprocessor import load_data
from tensorflow.data import Dataset
import tensorflow as tf
import json
import os

tf.executing_eagerly()

from picker_model import TargetModel


def image_mapping_fn(image_link):
    image_link = bytes.decode(image_link.numpy())
    return load_data(image_link)


class Trainer(TargetModel):
    def __init__(self, dataset=None, dataset_path=None):
        super().__init__()

        if dataset == None:
            dataset = self.read_from_dataset_path(dataset_path)
        self.dataset = dataset

        self.dataset_dict = {}
        for item in self.dataset:
            self.dataset_dict = {**self.dataset_dict, **item}

    def read_from_dataset_path(self, dataset_folder):
        json_files = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))

        data = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                data.append(json.load(f))
        return data

    def build_dataset(self):
        image_links = list(self.dataset_dict.keys())
        labels = list(self.dataset_dict.values())

        print(f"Создание датасета из {len(image_links)} изображений...")

        def process_image(image_link, label):
            image_tensor = tf.py_function(
                func=image_mapping_fn, inp=[image_link], Tout=tf.float32
            )
            image_tensor.set_shape(
                [None, None, None]
            )  

            return image_tensor, label
          
        dataset = Dataset.from_tensor_slices((image_links, labels))
        dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def train(self, epochs=10, learning_rate=0.001, validation_split=0.0):
        """Переопределяем метод train с нужными параметрами"""

        try:
            dataset = self.build_dataset()
            total_samples = len(self.dataset_dict)
            print(f"Общее количество образцов: {total_samples}")

            if total_samples == 0:
                print("Нет данных для обучения!")
                return None

            for batch in dataset.take(1):
                if isinstance(batch, tuple):
                    x_batch, y_batch = batch
                    print(f"X shape: {x_batch.shape}, Y shape: {y_batch.shape}")
                    print(f"X dtype: {x_batch.dtype}, Y dtype: {y_batch.dtype}")
                else:
                    print(f"Unexpected batch structure: {type(batch)}")
                    print(
                        f"Batch keys: {batch.keys() if hasattr(batch, 'keys') else 'No keys'}"
                    )

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            history = self.model.fit(
                dataset,
                epochs=epochs,
                verbose=1,
            )

            return history

        except Exception as e:
            print(f"Ошибка во время обучения: {e}")
            import traceback

            traceback.print_exc()
            return None

    def save_weights(self, filepath):
        """Сохраняет веса модели"""
        try:
            if hasattr(self, "model") and self.model is not None:
                self.model.save_weights(filepath)
                print(f"Веса модели сохранены в {filepath}")
            else:
                print("Модель не инициализирована для сохранения весов")
        except Exception as e:
            print(f"Ошибка при сохранении весов: {e}")

    def load_weights(self, filepath):
        """Загружает веса модели"""
        try:
            if hasattr(self, "model") and self.model is not None:
                self.model.load_weights(filepath)
                print(f"Веса модели загружены из {filepath}")
            else:
                print("Модель не инициализирована для загрузки весов")
        except Exception as e:
            print(f"Ошибка при загрузке весов: {e}")
