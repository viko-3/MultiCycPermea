import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor
import cairosvg
import os

# input your csv
path = ""
csv_file =  path + "*.csv"
df = pd.read_csv(csv_file)

image_folder = path + 'generate_images'
os.makedirs(image_folder, exist_ok=True)

rdDepictor.SetPreferCoordGen(True)
for index, row in df.iterrows():
    smiles = row['SMILES']
    image_path = os.path.join(image_folder, f"{row['CycPeptMPDB_ID']}.png")

    mol = Chem.MolFromSmiles(smiles)
    rdCoordGen.AddCoords(mol)

    view = Draw.rdMolDraw2D.MolDraw2DSVG(600, 600)
    view.DrawMolecule(Draw.rdMolDraw2D.PrepareMolForDrawing(mol))
    view.FinishDrawing()
    svg = view.GetDrawingText()

    svg_file_path = f"./temp_molecule_{index}.svg"
    with open(svg_file_path, "w") as f:
        f.write(svg)

    cairosvg.svg2png(url=svg_file_path, write_to=image_path)

    os.remove(svg_file_path)

df['image_path'] = df['CycPeptMPDB_ID'].apply(lambda x: os.path.join(image_folder, f"{x}.png"))
df.to_csv(csv_file, index=False)
