# flu
Generative models for flu hemagglutinin.
King Lab, Institute for Protein Design, University of Washington.

## Usage
First, run `data_processing.py` to do some pre-processing on the flu sequences.
Then, run `generator.py` to train a model and predictor and print sequences tuned for a specific property.

## Notes
This model implements batches and batch normalization in a sloppy/bad way. Batch sizes are *not* flexible once they are chosen. Also, batch normalization layers still use batch means and variances even during inference. These two errors do somewhat cancel out though: when using models for inference, we'll only care about the first sample in the batch. The rest will be random samples along for the ride andwill supply a reasonable estimate of means and variances during batch normalization.

This should be fixed someday. 
