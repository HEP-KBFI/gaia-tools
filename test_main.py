import unittest

from gaia_tools.data_analysis import get_transformed_data
from gaia_tools.import_functions import import_data


class gaiaTests(unittest.TestCase):
    """Tests for GeekTechStuff Grafana API Python"""

    def test_source_id_is_int(self):

        
        data_icrs = import_data("spectroscopic_test_table.csv")
        
        galcen_data = get_transformed_data(data_icrs, 
                                            include_cylindrical = True,
                                            debug = True,
                                            is_bayes = False, 
                                            is_source_included = True)

        self.assertIs(galcen_data.source_id.iloc[0], int)

    def test_source_id_is_match(self):

        data_icrs = import_data("spectroscopic_test_table.csv")
        
        galcen_data = get_transformed_data(data_icrs, 
                                            include_cylindrical = True,
                                            debug = True,
                                            is_bayes = False, 
                                            is_source_included = True)

        self.assertEqual(galcen_data.source_id.iloc[0], data_icrs.source_id.iloc[0])

    # TODO: Test if PHI parameter is within allowed range    
    def test_phi_is_valid(self):
        pass

    # TODO: 

    
    
if __name__ == '__main__':
    unittest.main()