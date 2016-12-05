from keras.callbacks import Callback

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.stats = {
            "train_loss": [],
            "batch_accuracy": [],
            "train_acc_history": [],
            "val_acc_history": []
        }

    def on_batch_end(self, batch, logs={}):
        self.stats["train_loss"].append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs={}):
        self.stats["train_acc_history"].append(logs.get("acc") * 100)
        self.stats["val_acc_history"].append(logs.get("val_acc") * 100)