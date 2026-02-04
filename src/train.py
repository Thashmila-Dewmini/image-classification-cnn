from data_loader import load_data
from model import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint  
import numpy as np
from sklearn.metrics import classification_report

X_train, y_train, X_test, y_test = load_data()

model = build_model()
model.summary()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("models/cnn_best_v1.keras", save_best_only=True)
]

# Train
model.fit(
    X_train,
    y_train, 
    batch_size=64, # uses 64 images at a time before updating weights
    epochs=30,
    validation_split=0.2,
    callbacks=callbacks
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# model.predict() : returns probabilities for all 10 classes
# argmax(axis=1) : picks the class with highest probability 
y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(y_test, y_pred))