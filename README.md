# LUNA
Learning UNiverse with Artificial intelligence (LUNA). Galaxy central Dark Matter with Random Forest
- The code for the paper

# The func inside func.py
- The packages to use: `sklearn`,`h5py`,`scipy`, `pandas`,`numpy`,`matplotlib`
## initilize
- read the data
  - From the IllustrisTNG 
    - `get_data`: read the GroupCat (`.hdf` files), return two dfs (data+Fir_ID)
    - `get_data_insimu`: zip the data and Fir_ID into one df
  - From the mock 
    - `get_data_inmock`: get the data from TangLin's catalogue (`.txt` files), return one df
    - `read_mock`: get the data from the mock (it's better to change into initialize)
- Initialize the data
  - `initialize`:
  - `read_mock`: 
## main functions
- matched the catalogue between IllustrisTNG(simu) and its mock observation(mock)
  - `main_compare_mock_simu`: The main func inside is `get_cross_gal`

## draw the plot
- `get_compare_plot_advance`
