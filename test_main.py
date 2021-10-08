import unittest

from gaia_tools.data_analysis import get_transformed_data
from gaia_tools.import_functions import import_data
import numpy as np


class gaiaTests(unittest.TestCase):
    

    def test_source_id_is_int(self):

        
        data_icrs = import_data("spectroscopic_test_table.csv")
        
        galcen_data = get_transformed_data(data_icrs, 
                                            include_cylindrical = True,
                                            debug = True,
                                            is_bayes = False, 
                                            is_source_included = True)

        self.assertIs(galcen_data.source_id.iloc[0], np.int64)

    def test_source_id_is_match(self):

        data_icrs = import_data("spectroscopic_test_table.csv")
        
        galcen_data = get_transformed_data(data_icrs, 
                                            include_cylindrical = True,
                                            debug = True,
                                            is_bayes = False, 
                                            is_source_included = True)

        self.assertEqual(galcen_data.source_id.iloc[0], data_icrs.source_id.iloc[0])

    # TODO: Test if PHI parameter is within allowed range at edge cases    
    def test_phi_is_valid(self):
        pass

    # [Using Astropy]
    # TODO: Test if x, y, z, v_x, v_y, v_z are > 0 or < 0 in a known case

    # [Using Astropy]
    # TODO: Test if r, phi, z, v_r, v_phi, v_z are > 0 or < 0 in a known case

    
    
    
if __name__ == '__main__':
    unittest.main()