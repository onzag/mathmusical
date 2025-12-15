# MathMusical V2

Based on an ancient version of mathmusical, this time using neural networks.

The purpose of this codebase is to generate general purpose musical structures based on the conception of musical shapes that exist within music; in the previous version the notes were generated at random in a modular space; and then transformed with a "what may go next approach" and a solution picked on what was most likely, but it used heuristics and not NN; this solution will use neural networks for increased accuracy and a training set, but remain working within the rules of the modular space, therefore generating variations.

## Curate

(Reference) MathMusical: Key and Chord Numerical Analysis for Symbolic Music.

The curate folder is the first step in processing the aria and lakh dataset, for the datasets https://github.com/loubbrad/aria-midi this shows the aria dataset, the pruned model is the one that was used here; for lahk https://colinraffel.com/projects/lmd/#get the full model was used.

`python curate/curate.py file.mid [dataset] [action]`

 - dataset: aria, lakh
 - action: curate, curate-ariafy

A curated midi file has a particular shape with specific tracks that give information on the piece, aria files however have no information on tempo and timings.

The curate ariafy action is available for the lakh dataset which makes a lahk file be more like an aria file (basically removes tempo and timing information), this is however not useful for most data processing; it's used mostly when developing to see patterns in notation software.

When you run the command directly it will open an interactive notation view.

The curate command in this codebase has not been optimized, but this purely numerical processing should be quick enough for that to matter.

To curate a whole folder run

`python curate/curate-dataset.py path/to/folder path/to/folder/curated [dataset]`

You must specify whether aria or lahk

## Expand

It will expand any curated midi file into all its forms, that is all the pitches it can take; this effectively multiplies any potential dataset size by 12; it is recommended to do this only with the lakh dataset first; as it is information complete, therefore the expansion shows its final midi form (but not from other metadata).

`python expand/expand.py curated-file.mid`

This will create a folder with the same name as the file and containing the files 0.mid, 1.mid, 2.mid all the way to 11

To expand a whole dataset run

`python expand/expand-dataset.py path/to/curated`

## Train the Neural Network for preprocessing Aria Files

(Reference) MathMusical: ...

Aria curated files need to have their timing information obtained so they are information complete, therefore a neural network is trained on the expanded lakh dataset

## Process Aria Files to be Information Complete

## Classify Aria Files (from its original dataset information)

## Train the Neural Network to classify Lahk Files

Lahk files lack information on their style, so in this scenario we used the processed information complete aria dataset to train a classifier that we will use upon the information complete lahk dataset

## Process and Classify Lahk Files

## Add misc information and other metadata

Extra information is added which is used to train the final neural network

## Dedupe the combined dataset

## Train the Generative neural network on the combined dataset; of expanded, information complete and classified midi files.

## Use the model

### Generate Base Template

#### Modifying Base Template

### Generate Base Harmony from Template

#### Modifiying Base Harmony

### Generate Notes on the Harmony

#### Inpainting musical notes

## Future Considerations

### Use the Lakh dataset to predict independent instruments and predict orchestation