# matched filter and upsampling signals

Our data capture application creates sample files with the following filename template: `fileprefix_freq_data_time.dat` e.g. `exp1_433.750M_2020-06-11_18-45-10.000.dat`

- *mixing_tools.py:* Library of functions used to import and process SDR data
- *sdr_data_processor.py:* given a single `.dat` file, it generates a matched filtered output 
- *Process SDR Data Directory:* Processes all sample data in a directory (assuming the files are named in a particular format) and stores the output in another directiory
- *matched filter upsampling:* Find the interpolated response of a matched filtered narrow band chirp signal