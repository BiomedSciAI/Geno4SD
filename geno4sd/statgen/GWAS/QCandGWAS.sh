#!/bin/bash
############################################################
############################################################
#
#		Author:  Aritra Bose, Computational Genomics
#				 IBM Research, Yorktown Heights, NY
#
#       Contact: a.bose@ibm.com
#
#  Last Update:  1/4/2023
#
#		README:  This runs a QC pipeline on GWAS data
#			     You can control whether you want to do 
#				 a PRS analysis or simple GWAS. If you 
# 				 are doing PRS then please comment the part
#				 titled for PRS. 
#
#				 The code generates a QC_logfile.txt which 
# 				 lists all the intermediate steps and log. 
#				 These can be referred to later for info
#				 about number of SNPs and individuals
#                filtered in each step. 
#
############################################################
############################################################
# 			DEFINING PATHS TO PACKAGES AND DATA
############################################################
############################################################

PATH_TO_PLINK=unset
PATH_TO_DATA=unset
OUT_PATH=unset
PROJ_NAME=unset
PATH_TO_GWA_SCRIPTS=unset
PATH_TO_SCRIPTS=unset
PATH_TO_MKL=unset
PATH_TO_TERAPCA=unset

usage()
{
    echo "Usage: GenoQC [ -p | --plink ] [ -d | --data ]
                        [ -o | --output ] [ -n | --project_name ]
                        [ -g | --scripts1 ] [ -s | --scripts2 ]
                        [ -t | --mkl ] [ -t | -terapca ]"
    exit 2
}

SHORT=p:,d:,o:,n:,g:,s:,m:,t:,h
LONG=plink:,data:,output:,project_name:,scripts1:,scripts2:,mkl:,terapca:,help
OPTS=$(getopt -a -n qc --options $SHORT --longoptions $LONG -- "$@")
VALID_ARGUMENTS=$?

if [ "$VALID_ARGUMENTS" != "0" ]; then
    usage
fi

echo "PARSED ARGUMENTS are $OPTS"
eval set -- "$OPTS"

while :
do
    case "$1" in
        -p | --plink )
            PATH_TO_PLINK="$2"
            shift 2 
            ;;
        -d | --data )
            PATH_TO_DATA="$2"
            shift 2 
            ;;
        -o | --output )
            OUT_PATH="$2"
            shift 2
            ;;
        -n | --project_name )
            PROJ_NAME="$2"
            shift 2
            ;;
        -g | --scripts1 )
            PATH_TO_GWA_SCRIPTS="$2"
            shift 2
            ;;
        -s | --scripts2 )
            PATH_TO_SCRIPTS="$2"
            shift 2
            ;;
        -m | --mkl )
            PATH_TO_MKL="$2"
            shift 2 
            ;;
        -t | --terapca )
            PATH_TO_TERAPCA="$2"
            shift 2
            ;;
        -h | --help )
            usage
            exit 2
            ;;
        --)
            shift;
            break
            ;;
        *)
            echo "Unexpected option :$1"
            usage
            ;;
            
     esac
done

echo .
echo "=========================="
echo "SETTING PATH VARIABLES"
echo "=========================="
echo "PATH_TO_PLINK : $PATH_TO_PLINK" 
echo "PATH_TO_DATA : $PATH_TO_DATA"
echo "OUT_PATH : $OUT_PATH"
echo "PROJ_NAME : $PROJ_NAME"
echo "PATH_TO_GWA_SCRIPTS : $PATH_TO_GWA_SCRIPTS"
echo "PATH_TO_SCRIPTS : $PATH_TO_SCRIPTS"
echo "PATH_TO_MKL : $PATH_TO_MKL"
echo "PATH_TO_TERAPCA : $PATH_TO_TERAPCA"
echo "=========================="
echo "=========================="
echo .
############################################################
######################	 MISSINGNESS   ######################	
############################################################

"$PATH_TO_PLINK"/plink2 --pfile "$PATH_TO_DATA" --geno 0.02 --mind 0.02 --make-pgen --out "$OUT_PATH"/qc1

cat "$OUT_PATH"/qc1.log >> "$PROJ_NAME"_QC_logfile.txt 
echo "################################################" >> "$PROJ_NAME"_QC_logfile.txt

############################################################
######################	 SEX CHECK   #######################	
############################################################

## extract X Chromosome snps
awk '{if($1 == "X") print $3}' "$PATH_TO_DATA".pvar > "$OUT_PATH"/chrX_snps.txt

## make a genotype file of X chromosome
"$PATH_TO_PLINK"/plink2 --pfile "$OUT_PATH"/qc1 --extract "$OUT_PATH"/chrX_snps.txt --make-pgen --out "$OUT_PATH"/"$PROJ_NAME"_chrX

"$PATH_TO_PLINK"/plink --bfile "$OUT_PATH"/"$PROJ_NAME"_chrX --check-sex

grep "PROBLEM" plink.sexcheck | awk '{print $1,$2}' > "$OUT_PATH"/sex_discrepancy.txt

"$PATH_TO_PLINK"/plink2 --pfile "$OUT_PATH"/qc1 --remove "$OUT_PATH"/sex_discrepancy.txt --make-bed --out "$OUT_PATH"/qc2

cat "$OUT_PATH"/qc2.log >> "$PROJ_NAME"_QC_logfile.txt 

rm "$OUT_PATH"/qc1.* 
rm R_check*


############################################################
######################		 MAF      ######################	
############################################################

#Select autosomal SNPs only (i.e., from chromosomes 1 to 22).
awk '{ if ($1 >= 1 && $1 <= 22) print $2 }' "$OUT_PATH"/qc2.bim > "$OUT_PATH"/snp_1_22.txt

"$PATH_TO_PLINK"/plink2 --bfile "$OUT_PATH"/qc2 --extract "$OUT_PATH"/snp_1_22.txt --make-pgen --out "$OUT_PATH"/qc3

rm "$OUT_PATH"/qc2.* 

"$PATH_TO_PLINK"/plink2 --pfile "$OUT_PATH"/qc3 --maf 0.05 --make-bed --out "$OUT_PATH"/qc4

cat "$OUT_PATH"/qc4.log >> "$PROJ_NAME"_QC_logfile.txt

rm "$OUT_PATH"/qc3.* 

############################################################
######################		 HWE      ######################	
############################################################


#HWE filter for controls

"$PATH_TO_PLINK"/plink2 --bfile "$OUT_PATH"/qc4 --hwe 1e-6 --make-bed --out "$OUT_PATH"/qc5 

cat "$OUT_PATH"/qc5.log >> "$PROJ_NAME"_QC_logfile.txt

rm "$OUT_PATH"/qc4.*

"$PATH_TO_PLINK"/plink --bfile "$OUT_PATH"/qc5 --hwe 1e-10 --hwe-all --make-bed --out "$OUT_PATH"/qc6 

cat "$OUT_PATH"/qc6.log >> "$PROJ_NAME"_QC_logfile.txt

rm "$OUT_PATH"/qc5.*

############################################################
################ 	 Heterozygosity Rate   #################	
############################################################

"$PATH_TO_PLINK"/plink --bfile "$OUT_PATH"/qc6 --exclude "$PATH_TO_GWA_SCRIPTS"/inversion.txt --range --indep-pairwise 250 50 0.25 --out "$OUT_PATH"/indepSNP

"$PATH_TO_PLINK"/plink --bfile "$OUT_PATH"/qc6 --extract "$OUT_PATH"/indepSNP.prune.in --het --out R_check

Rscript --no-save "$PATH_TO_GWA_SCRIPTS"/heterozygosity_outliers_list.R 

sed 's/"// g' fail-het-qc.txt | awk '{print$1, $2}'> "$OUT_PATH"/het_fail_ind.txt

"$PATH_TO_PLINK"/plink2 --bfile "$OUT_PATH"/qc6 --remove "$OUT_PATH"/het_fail_ind.txt --make-bed --out "$OUT_PATH"/qc7 

cat "$OUT_PATH"/qc7.log >> "$PROJ_NAME"_QC_logfile.txt

rm "$OUT_PATH"/qc6.*
rm fail-het-qc.txt

############################################################
################ 	 Cryptic Relatedness   #################	
############################################################

"$PATH_TO_PLINK"/plink --bfile "$OUT_PATH"/qc7 --extract "$OUT_PATH"/indepSNP.prune.in --genome --min 0.2 --out "$OUT_PATH"/pihat_min0.2

awk '{ if ($8 >0.9) print $0 }' "$OUT_PATH"/pihat_min0.2.genome > "$OUT_PATH"/zoom_pihat.genome

"$PATH_TO_PLINK"/plink2 --bfile "$OUT_PATH"/qc7 --missing --out "$OUT_PATH"/qc7

awk '{print $2,"\t"$4}' "$OUT_PATH"/pihat_min0.2.genome > "$OUT_PATH"/pihat_min0.2_ids.txt

#this will create a file with IDs to remove
python3 "$PATH_TO_SCRIPTS"/makeibdfam.py "$OUT_PATH"/pihat_min0.2_ids.txt "$OUT_PATH"/qc7.smiss

awk '{print $1,$1}' IBD_samples_toremove.txt | sed 1d > "$OUT_PATH"/ibdtorem.txt

#Populate individuals with lowest call rate in ibdtorem.txt and remove them
"$PATH_TO_PLINK"/plink2 --bfile "$OUT_PATH"/qc7 --remove "$OUT_PATH"/ibdtorem.txt --make-bed --out "$OUT_PATH"/qc_pre_final

cat "$OUT_PATH"/qc_pre_final.log >> "$PROJ_NAME"_QC_logfile.txt

echo "################################################" >> "$PROJ_NAME"_QC_logfile.txt
rm "$OUT_PATH"/qc7.*
rm IBD_samples_toremove.txt

############################################################
############     Remove multiallelic SNPs   ################
############################################################

python3 "$PATH_TO_SCRIPTS"/getbiallelic.py "$OUT_PATH"/qc_pre_final.bim

sed 1d biallelicsnps.txt > tmp
mv tmp biallelicsnps.txt

"$PATH_TO_PLINK"/plink2 --bfile "$OUT_PATH"/qc_pre_final --extract biallelicsnps.txt --make-bed --out "$OUT_PATH"/qc_final

cat "$OUT_PATH"/qc_final.log >> "$PROJ_NAME"_QC_logfile.txt
rm biallelicsnps.txt

echo "################################################" >> "$PROJ_NAME"_QC_logfile.txt

#################################################
############     Prune for PCA   ################
#################################################

# LD prune
"$PATH_TO_PLINK"/plink --bfile "$OUT_PATH"/qc_final --exclude "$PATH_TO_GWA_SCRIPTS"/inversion.txt --range --indep-pairwise 200 50 0.25 --out "$OUT_PATH"/indepSNP

mkdir "$OUT_PATH"/PopStrat

#LD pruned data set for PCA
"$PATH_TO_PLINK"/plink2 --bfile "$OUT_PATH"/qc_final --extract "$OUT_PATH"/indepSNP.prune.in --make-bed --out "$OUT_PATH"/PopStrat/ukbimp_ps_1

cat "$OUT_PATH"/PopStrat/ukbimp_ps_1.log >> "$PROJ_NAME"_QC_logfile.txt

echo "#################################################" >> "$PROJ_NAME"_QC_logfile.txt
echo "#########   Performed Quality Control  ########## " >> "$PROJ_NAME"_QC_logfile.txt
echo "#################################################" >> "$PROJ_NAME"_QC_logfile.txt

#################################################
############     PCA   ################
#################################################

#set environment variables for TeraPCA
source "$PATH_TO_MKL"

"$PATH_TO_TERAPCA" -bfile "$OUT_PATH"/PopStrat/ukbimp_ps_1 -filewrite 1 -prefix "$OUT_PATH"/"$PROJ_NAME" -memory 2 -nsv 50

echo " "
echo "#################################################" >> "$PROJ_NAME"_QC_logfile.txt
echo "#########         Performed PCA        ########## " >> "$PROJ_NAME"_QC_logfile.txt
echo "#################################################" >> "$PROJ_NAME"_QC_logfile.txt