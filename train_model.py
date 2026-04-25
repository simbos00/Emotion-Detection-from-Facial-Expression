import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import kagglehub

def build_net(optim, input_shape=(48, 48, 1), num_classes=7):
    """
    Constructs the Deep Convolutional Neural Network
    """
    model = models.Sequential(name='FER_CNN')

    # --- Preprocessing & Augmentation ---
    # Rescaling 1./255 normalizes inputs to [0,1]
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.04)) 
    model.add(layers.RandomZoom(0.15))
    model.add(layers.RandomTranslation(height_factor=0.15, width_factor=0.15))

    # --- Block 1: 64 Filters, 3x3 Kernel ---
    # Conv2d(1, 64, 3, padding=1) -> eLU -> BatchNorm -> MaxPool(2) -> Dropout(0.25)
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='elu', 
                            kernel_initializer='he_normal', name='conv1'))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='pool1'))
    model.add(layers.Dropout(0.25, name='drop1'))

    # --- Block 2: 128 Filters, 5x5 Kernel ---
    # Conv2d(64, 128, 5, padding=2) -> eLU -> BatchNorm -> MaxPool(2) -> Dropout(0.25)
    model.add(layers.Conv2D(128, (5,5), padding='same', activation='elu', 
                            kernel_initializer='he_normal', name='conv2'))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='pool2'))
    model.add(layers.Dropout(0.25, name='drop2'))

    # --- Block 3: 256 Filters, 3x3 Kernel ---
    # Conv2d(128, 256, 3, padding=1) -> eLU -> BatchNorm -> MaxPool(2) -> Dropout(0.25)
    model.add(layers.Conv2D(256, (3,3), padding='same', activation='elu', 
                            kernel_initializer='he_normal', name='conv3'))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='pool3'))
    model.add(layers.Dropout(0.25, name='drop3'))

    # --- Block 4: 512 Filters, 3x3 Kernel ---
    # Conv2d(256, 512, 3, padding=1) -> eLU -> BatchNorm -> MaxPool(2) -> Dropout(0.25)
    model.add(layers.Conv2D(512, (3,3), padding='same', activation='elu', 
                            kernel_initializer='he_normal', name='conv4'))
    model.add(layers.BatchNormalization(name='bn4'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='pool4'))
    model.add(layers.Dropout(0.25, name='drop4'))

    # --- Classification Layers ---
    model.add(layers.Flatten(name='flatten'))
    
    # Linear(512 * 3 * 3, 256) -> eLU -> BatchNorm -> Dropout(0.25)
    model.add(layers.Dense(256, activation='elu', kernel_initializer='he_normal', name='dense1'))
    model.add(layers.BatchNormalization(name='bn_dense1'))
    model.add(layers.Dropout(0.25, name='drop_dense1'))
    
    # Linear(256, 512) -> eLU -> BatchNorm -> Dropout(0.25)
    model.add(layers.Dense(512, activation='elu', kernel_initializer='he_normal', name='dense2'))
    model.add(layers.BatchNormalization(name='bn_dense2'))
    model.add(layers.Dropout(0.25, name='drop_dense2'))
    
    # Linear(512, NUM_CLASSES)
    model.add(layers.Dense(num_classes, activation='softmax', name='out_layer'))

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    """
    Generates and saves plots for accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    
    # Save the figure
    save_path = 'training_plots.png'
    plt.savefig(save_path)
    print(f"\nPlots saved to {save_path}")
    
    # Show the figure
    plt.show()

if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")

    # --- 1. DOWNLOAD & SETUP DATA ---
    # define the remote kaggle directory with the data
    path = kagglehub.dataset_download("msambare/fer2013")
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')
    print(train_dir, test_dir)
    # train_dir, test_dir = setup_kaggle_data() # use this if you want to download the dataset

    # --- Configuration ---
    img_size = 48
    batch_size = 32
    epochs = 100

    # --- Data Loading ---
    print(f"Loading training data from: {train_dir}")
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='categorical',
        shuffle=True
    )

    print(f"Loading validation data from: {test_dir}")
    val_ds = image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        label_mode='categorical',
        shuffle=False
    )

    # Optimize performance 
    # (autotune finds the best buffer size dynamically)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- Build and Train Model ---

    # 2. Define Optimizer separate from the model build
    opt = optimizers.Nadam(
        learning_rate=0.001, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-07, 
        name='Nadam'
    )

    # 3. Build the Net
    emotion_model = build_net(optim=opt, input_shape=(img_size, img_size, 1), num_classes=7)
    emotion_model.summary()

    # 4. Define callbacks 
    callbacks = [
        ModelCheckpoint(
            "emotion_model_weights.keras", 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.5, 
            patience=7, 
            verbose=1, 
            min_lr=1e-7
        ),
        EarlyStopping(
            monitor='val_accuracy', 
            min_delta=0.00005,
            patience=11, 
            restore_best_weights=True, 
            verbose=1
        )
    ]

    print("\nStarting training...")
    try:
        history = emotion_model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )
        print("\nTraining completed. Best weights saved as 'emotion_model_weights.keras'.")
        
        # --- PLOTTING ---
        plot_training_history(history)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")