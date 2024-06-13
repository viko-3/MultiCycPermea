# MultiCycPermea

### Data
We supply the split csv_file, it include SMILES of cyclic peptide. As for image, we supply the code (draw_peptide_images.py) to draw.

### Train
```python
CUDA_VISIBLE_DEVICES={n}  python main.py --use_image_info True/False --feature_cmb_type concate
```
### Test
It same as train， you can change the setting in test.py.

### You can download the pretrained image_encoder and text_encoder are in Zendo.
### We also supply the preprocessed substructure knowledge graph in Zendo.
