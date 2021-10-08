import unittest

from gaia_tools.data_analysis import*
from gaia_tools.import_functions import *


class gaiaTests(unittest.TestCase):
    """Tests for GeekTechStuff Grafana API Python"""

    def test_source_id_is_int(self):

        print("Echo")
        #data_icrs = import_data("spectroscopic_test_table.csv")
        
        # galcen_data = get_transformed_data(data_icrs, 
        #                                     include_cylindrical = True,
        #                                     debug = True,
        #                                     is_bayes = False, 
        #                                     is_source_included = True)

        #self.assertIs(galcen_data.source_id.iloc[0], int)


    
    
if __name__ == '__main__':
    unittest.main()