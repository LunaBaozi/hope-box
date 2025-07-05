from rdkit import Chem
from syba.syba import SybaClassifier

syba = SybaClassifier()
syba.fitDefaultScore()
smi = "O=C(C)Oc1ccccc1C(=O)O"
print(syba.predict(smi))
# syba works also with RDKit RDMol objects
mol = Chem.MolFromSmiles(smi)
syba.predict(mol=mol)
# syba.predict is actually method with two keyword parameters "smi" and "mol", if both provided score is calculated for compound defined in "smi" parameter has the priority
syba.predict(smi=smi, mol=mol)