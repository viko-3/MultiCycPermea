
I create several column, each meaning shown as follows:

1.cliff_split_id: each mol has its own unique cliff_split_id and saved in cliff_split_id column, it starts from 0. !!!!warning :Training set and test set cannot be splited by this id. It can only be used for data analysis.

2.is_cliff: this column is used to tell whether this peptide has cliff partner


3.cliff_to_which_array: if a peptide has one or more cliff partner, this column will contain an array about the specific partner information (cliff_split_id) 
