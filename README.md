# DeepnsSNPs

DeepnsSNPs:
Many diseases of genetic origin originate from non-synonymous single nucleotide polymorphisms (nsSNPs). These cause changes in the final protein product encoded by a gene. Through large scale sequencing and population studies, there is growing availability of information on which variations are tolerated and which are not. Variant effect predictors use a wide range of information about such variations to predict their effect, often focusing on evolutionary information. Herein, a novel amino acid substitution variant effect predictor is developed. The predictor is a deep convolutional neural network incorporating evolutionary information, residue environment information of mutant site, and protein predicted structural information to predict both the pathogenicity and the severity of amino acid substitutions. The model achieves excellent performance on two benchmark datasets. 


## Two benchmark datasets 
### 1. PredictSNP dataset 
The training datasets of all evaluated prediction tools were collected, and mutations overlapping between the training datasets and our dataset were removed to create a fully independent PredictSNP benchmark dataset. All selected prediction tools use at least some position-specific parameters derived from evolutionary information as significant pathogenicity indicators. Therefore, the author of PredicteSNP removed both directly overlapping mutations and mutations at any overlapping positions, i.e., positions that were mutated in the training datasets of selected prediction tools. The positions were considered overlapping if they were located in the fragments of two sequences aligned by BLAST search with e-value 10-10, and the aligned fragments had at least 50% identity. Finally, all mutations at positions overlapping with testing datasets were removed to assure independence between the PredictSNP benchmark and the testing datasets (MMP dataset). As a complement to the independent PredictSNP benchmark dataset.
 
### 2. MMP dataset
The MMP dataset was compiled from experimental studies summarized by Yampolsky and Stoltzfus. One protein out of twelve was excluded due to its very short length (30 amino acid residues). This set was complemented by mutations gathered from two patent applications issued by Danisco Inc. describing the effect of mutations on serine protease from Bacillus subtilis and alpha-amylase from Geobacillus stearothermophilus. This dataset of thirteen massively mutated proteins (MMP) initially contained 16,500 mutations. For all mutations originating from experimental studies, a change of activity is provided in the form of a categorical value. In patent applications, the specific ratio of activity change for each mutation is known, and according to the information enclosed in the source materials, the mutations with the activity changes larger than 50% were considered deleterious. Finally, the mutations at the positions overlapping with the evaluated tools' training datasets were removed from the MMP dataset. 

More details information about the PredictSNP and MMP datasets, please refer to: Bendl J, Stourac J, Salanda O, et al. PredictSNP: Robust and Accurate Consensus Classifier for Prediction of Disease-Related Mutations [J]. PLOS Computational Biology, 2014, 10(1). 



## Extracting feature matrix
### 1. Position Specific Score Matrix (PSSM)
We adopted the PSI-BLAST (Position-Specific Iterated Basic Local Alignment Search Tool) to generate PSSM information. For more detail information, please refer to: 
Schaffer A A, Aravind L, Madden T L, et al. Improving the accuracy of PSI-BLAST protein database searches with composition-based statistics and other refinements [J]. Nucleic Acids Research, 2001, 29(14): 2994-3005.  
Software website: https://www.ebi.ac.uk/seqdb/confluence/display/THD/PSI-BLAST.

### 2.Predicted Secondary Structure (PSS)
We applied the PSIPRED software to extract SS characteristics. For each query protein sequence with L residues, the outputs of PSIPRED is the probability matrix with L×3 dimension. 
For more detail information, please refer to: 
Mcguffin L J, Bryson K, Jones D T, et al. The PSIPRED protein structure prediction server[J]. Bioinformatics, 2000, 16(4): 404-405, and, https://hpc.nih.gov/apps/PSIPRED.html.

### 3. Predicted Protein Relative Solvent Accessibility (PRSA) 
We apply the SANN program to generate PRSA. Ultimately, we could get the PRSA matrix with L×3 dimension for each sequence. In the PRSA matrix, each row is composed of three values, which indicate the probability of one residue belonging to three RSA classes (i.e., exposed, intermediate, and hidden). For more detailed information about SANN software, please refer to: Joo K, Lee S J, Lee J, et al. Sann: Solvent accessibility prediction of proteins by nearest neighbor method[J]. Proteins, 2012, 80(7): 1791-1797

### 4. Disorder
Dynamically disordered regions appear to be relatively abundant in eukaryotic proteomes. The DISOPRED server allows users to submit a protein sequence and returns a probability estimate of each residue in the sequence being disordered. The results are sent in both plain text and graphical formats, and the server can also supply predictions of secondary structure to provide further structural information.
See more detail information, please refer to:
Jonathan J. Ward, Liam J. McGuffin, Kevin Bryson, Bernard F. Buxton, David T. Jones, The DISOPRED server for the prediction of protein disorder, Bioinformatics, Volume 20, Issue 13, 1 September 2004, Pages 2138–2139, https://doi.org/10.1093/bioinformatics/bth195


## Machine Learning Classifiers
In this work, we utilize four classifiers: Random Forest, Support Vector Machine, K-nearest Neighbor, and Decision Tree from sci-kit-learn, which is a "Simple and efficient tools for data mining and data analysis", "Accessible to everybody, and reusable in various contexts","Built on NumPy, SciPy, and matplotlib", and "Open source, commercially usable - BSD license". See more details from https://scikit-learn.org/stable/index.html

## Contact 
If you are interested in our work, OR, if you have any suggestions/questions about our work, PLEASE contact with us. 
E-mail: gfang0616@njust.edu.cn



