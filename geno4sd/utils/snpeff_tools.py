import os, re
import pandas as pd

# Basic parameters for the java command
mem_size = '16g'
rootdir = re.sub( 'Geno4SD/.*', 'Geno4SD', os.getcwd())
java_cmd = 'java -Xmx'+mem_size + ' -jar ' + os.path.join( rootdir, 'geno4sd', 'utils', 'snpEff', 'snpEff.jar')

def download_genome(genome):
    """
    Function to download genome
    
    :params genome: name of genome to download
    """
    os.system( java_cmd + ' download -v ' + genome )


def cancer_analysis(inputvcf, outputvcf, genome = 'GRCh38.p13'):
    """
    Function to run SnpEff cancer analysis annotation mode
    
    :params inputvcf: <string> path to input vcf
    :params outputvcf: <string> path to output vcf
    :params genome: <string> Genome version to use. Be sure of have already downloaded it. Default is "GRCh38.p13"
    """
    os.system( java_cmd + ' -v -cancer ' + genome + ' ' + inputvcf + ' > ' + outputvcf )


def parse_vcf(inputvcf):
    """
    Function to parse vcf into panda dataframe
    
    :params inputvcf: <string> path to input vcf
    :return: <dataframe> parsed object
    
    """
    
    header_index = 0    
    l = 0
    lines = []
    for line in open(inputvcf):
        l += 1
        line = line.strip()
        if ('#CHROM' in line) & (header_index == 0):
            header_index = l             
        if (header_index != 0):
            lines.append(line.split('\t'))
    
    df = pd.DataFrame( lines )
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:]
    df['INFO'] = [ x.split(';') for x in df['INFO'] ]
    
    return( df )