from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


def callback_binary_hate():

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_f1',  
                                           factor = 0.8,          
                                           patience = 2,         
                                           min_lr = 1e-6,        
                                           verbose = 0,
                                           mode='max')           

  early_stop = EarlyStopping(monitor = 'val_f1',       
                             patience = 3,                 
                             restore_best_weights = True,
                             verbose = 0,
                             mode='max')

  checkpoint = ModelCheckpoint('models/binary_hate/model_hate_binary.keras',
                               monitor = 'val_f1',
                               save_best_only = True,
                               save_weights_only = False,
                               verbose = 1,
                               mode='max')

  return early_stop, checkpoint, reduce_learning_rate

# ------------------------------

def callback_hate_type():

  reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',   
                                           factor = 0.8,           
                                           patience = 3,            
                                           min_lr = 1e-6,           
                                           verbose = 0)            

  early_stop = EarlyStopping(monitor = 'val_loss',         
                             patience = 10,               
                             restore_best_weights = True,  
                             verbose = 1)

  checkpoint = ModelCheckpoint(f'models/hate_type/model_hate_type.h5',
                               monitor = 'val_loss',
                               save_best_only = True,
                               save_weights_only = False,
                               verbose = 1)

  return early_stop, checkpoint, reduce_learning_rate