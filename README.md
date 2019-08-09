Generative models for flu hemagglutinin.
King Lab, Institute for Protein Design, University of Washington.

# Usage
First, run `data_processing.py` to do some pre-processing on the flu sequences.
Then, run `generator.py` to train a model and predictor and print sequences tuned for a specific property.
Use the command-line options:
```
OPTION                      DEFAULT VALUE     DESCRIPTION
  --data                       all              data to train on, one of "all", "h1", "h2", "h3", ..., "h18", or "aligned" (others are not aligned)
  --encoding                   categorical      data encoding, either "categorical" or "blosum"
  --model                      vae_fc           model to use, one of "gan", "vae_fc", "vae_conv", or "vae_lstm"
  --beta                       5                if using a VAE, the coefficient for the KL loss
  --tuner                      design           what to tune for, a combination of "subtype", "head_stem", or "design" (comma separated)
  --design                     1-M              if using design tuner, list of strings "[position]-[residue]-[weight]" (weight is optional), e.g. "15-R-1.0,223-C-5.0"
  --subtype                                     if using subtype tuner, which subtype you want
  --head_stem                                   if using head-stem tuner, a string of "[head subtype],[stem subtype]"
  --train_model_epochs         0                how many epochs to train the generative model
  --train_predictor_epochs     0                how many epochs to train the predictor model
  --tune_epochs                0                how many epochs to tune
  --batch_size                 100              batch size for training everything
  --latent_dimension           100              latent dimension for everything
  --restore_model                               saved file to restore model from
  --restore_predictor                           saved file to restore predictor from
  --save_model                                  where to save model to
  --save_predictor                              where to save predictor to
  --num_outputs                1                how many samples to print out
  --random_seed                                 random seed to make execution deterministic, default is random
  --return_latents                              1 if you want to print the latent variable with the sequence
  --channels                   16               number of channels in convolution hidden layers
  --reconstruct                                 if you want to pass a sequence through a VAE
  --print_from_latents                          print sequences from a comma separated list of latent variable arrays
```

# Example usages
*Train a convolutional VAE and save it for later* 
```
python generator.py \
    --model=vae_conv \
    --train_model_epochs=400 \
    --channels=32 \
    --latent_dimension=50 \
    --save_model=path/to/save/folder/
```
*Train a subtype predictor* 
```
python generator.py \
    --tuner=subtype \  
    --train_predictor_epochs=30 \
    --save_predictor=path/to/save/folder/
```
*Load an existing model and predictor and tune a sequence to be a H3 with a methionine at position 130* 
```
python generator.py \
    --model=vae_conv \
    --restore_model=path/to/save/folder/ \
    --tuner=design,subtype \
    --restore_predictor=path/to/save/folder2/ \
    --subtype=3 \
    --design=130-M \
    --tune_epochs=1000
```
*Generate 100 H1s* 
```
python generator.py \
    --model=vae_conv \
    --restore_model=path/to/save/folder/ \
    --tuner=subtype \
    --restore_predictor=path/to/save/folder2/ \
    --subtype=1 \
    --tune_epochs=1000 \
    --num_outputs=100
```

# Notes
Some options are incompatible. For example, if you specify a saved predictor with `--restore_predictor` but your `--subtype` option doesn't need a predictor, this will result in an error.
