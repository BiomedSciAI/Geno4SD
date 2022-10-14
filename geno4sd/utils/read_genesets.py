__author__ = "Kahn Rhrissorrakrai"
__copyright__ = "Copyright 2022, IBM Research"
__version__ = "0.0.1"
__maintainer__ = "Kahn Rhrissorrakrai"
__email__ = "krhriss@us.ibm.com"
__status__ = "Development"

# Load geneset file (gmt)
def readGMT( geneset ):
    """
    Function to read in GMT geneset file
    
    :param geneset: path to GMT file
    :return: dictionary of genesets with geneset name as keys and list of genes as values
    """
    file1 = open(geneset, 'r')
    Lines = file1.readlines()
  
    gsetMap = {}
    for line in Lines:
        line = line.strip()
        df = line.split('\t')
        gsetMap[ df[0] ] = df[2:len(df)]
    return gsetMap