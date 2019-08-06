Generative models for flu hemagglutinin.
King Lab, Institute for Protein Design, University of Washington.

# Usage
First, run `data_processing.py` to do some pre-processing on the flu sequences.
Then, run `generator.py` to train a model and predictor and print sequences tuned for a specific property.
Use the command-line options:
```
  --data                     data to train on, one of "all", "h1", "h2", "h3", ..., "h18", or "aligned" (others are not aligned)
  --encoding                 data encoding, either "categorical" or "blosum"
  --model                    model to use, one of "gan", "vae_fc", "vae_conv", or "vae_lstm"
  --beta                     if using a VAE, the coefficient for the KL loss
  --tuner                    what to tune for, a combination of "subtype", "head_stem", or "design" (comma separated)
  --design                   if using design tuner, list of strings "[position]-[residue]-[weight]" (weight is optional), e.g. "15-R-1.0,223-C-5.0"
  --subtype                  if using subtype tuner, which subtype you want
  --head_stem                if using head-stem tuner, a string of "[head subtype],[stem subtype]"
  --train_model_epochs       how many epochs to train the generative model
  --train_predictor_epochs   how many epochs to train the predictor model
  --tune_epochs              how many epochs to tune
  --batch_size               batch size for training everything
  --latent_dimension         latent dimension for everything
  --restore_model            saved file to restore model from
  --restore_predictor        saved file to restore predictor from
  --save_model               where to save model to
  --save_predictor           where to save predictor to
  --num_outputs              how many samples to print out
  --random_seed              random seed to make execution deterministic, default is random
  --return_latents           1 if you want to print the latent variable with the sequence
  --channels                 number of channels in convolution hidden layers
```
# Notes
The batch normalization uses a true mean to compute population statistics, so the totals might get large after a lot of training.

Some options are incompatible. For example, if you specify a saved predictor with `--restore_predictor` but your `--subtype` option doesn't need a predictor, this will result in an error.
