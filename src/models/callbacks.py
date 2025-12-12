from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


def callback_binary_hate():

  '''
  Builds and returns the training callbacks for the binary hate-speech classifier.
  The function configures three components: 
  (1) ReduceLROnPlateau, which decreases the learning rate when the validation loss stops improving;
  (2) EarlyStopping, which halts training once the model no longer progresses and restores the best weights; 
  (3) ModelCheckpoint, which saves the best-performing model based on validation loss.
  Outputs a tuple (early_stop, checkpoint, reduce_learning_rate) to be passed directly to model.fit().
  '''

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                                           factor = 0.8,
                                           patience = 4,
                                           min_lr = 1e-6,        
                                           verbose = 0)

  early_stop = EarlyStopping(monitor = 'val_loss',
                             patience = 10,
                             restore_best_weights = True,
                             verbose = 0)

  checkpoint = ModelCheckpoint('models/binary_hate/model_hate_binary.h5',
                               monitor = 'val_loss',
                               save_best_only = True,
                               save_weights_only = False,
                               verbose = 1)

  return early_stop, checkpoint, reduce_learning_rate

# ------------------------------

def callback_hate_type():

  '''
  Builds and returns the training callbacks for the hate type-speech classifier.
  The function configures three components: 
  (1) ReduceLROnPlateau, which decreases the learning rate when the validation loss stops improving;
  (2) EarlyStopping, which halts training once the model no longer progresses and restores the best weights; 
  (3) ModelCheckpoint, which saves the best-performing model based on validation loss.
  Outputs a tuple (early_stop, checkpoint, reduce_learning_rate) to be passed directly to model.fit().
  '''

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',   
                                           factor = 0.8,
                                           patience = 4,
                                           min_lr = 1e-6,
                                           verbose = 0)            

  early_stop = EarlyStopping(monitor = 'val_loss',
                             patience = 10,
                             restore_best_weights = True,
                             verbose = 0)

  checkpoint = ModelCheckpoint(f'models/hate_type/model_hate_type.h5',
                               monitor = 'val_loss',
                               save_best_only = True,
                               save_weights_only = False,
                               verbose = 1)

  return early_stop, checkpoint, reduce_learning_rate