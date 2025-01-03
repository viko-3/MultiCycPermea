# MultiCycPermea

### Data
We supply the split csv_file, it include SMILES of cyclic peptide. As for image, we supply the code (draw_peptide_images.py) to draw.

### Train
```python
CUDA_VISIBLE_DEVICES={n}  python main.py --use_image_info True/False --feature_cmb_type concate
```
### Test
It same as train， you can change the setting in test.py.

### You can download the pretrained image_encoder and text_encoder are in [here][dropbox-link]. 
### We also supply the preprocessed substructure knowledge graph in [here][dropbox-link].
[dropbox-link]: https://www.dropbox.com/scl/fo/bhl86a9cjjl6enweowola/APV6blUE2Q41_08M0l4fIJA?rlkey=cxvrkldylm03z6mtmuk6gatj5&st=q3esj07j&dl=0
