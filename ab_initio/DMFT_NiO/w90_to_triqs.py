from triqs_dft_tools.converters.wannier90 import Wannier90Converter
from h5 import *
                
def w90converter(input_converter_file):
    Converter = Wannier90Converter(seedname=input_converter_file, hdf_filename='w90_to_triqs.h5')    # threshold !!! , w90zero=2e-03
    Converter.convert_dft_input()
    print("Conversion Done")

###########################################################################################
###########################           Parameters             ##############################
###########################################################################################

input_converter_file = 'NiO'

################################################################²###########################
##############################           Main             #################################
###########################################################################################

w90converter(input_converter_file)
