# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:02:39 2024

@author: Vero
"""
# %%

import fgs

# %%

import pandas as pd

# Se importan los datos.

ruta_archivo = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\fichero compuestos.csv'

compuestos = pd.read_csv(ruta_archivo, sep=';')


# %%

# Se extrae la estructura “parent” o principal del complejo de moléculas.

import rdkit

from rdkit import Chem

from rdkit.Chem import inchi

from chembl_structure_pipeline import standardizer as sdz

 

compuestos["pmol"] = compuestos.inchi.apply(lambda x: sdz.get_parent_mol(Chem.MolFromInchi(x))[0])


# %%

from chembl_structure_pipeline import standardizer as sdz

# Una vez obtenidas las estructuras "parent" se estandarizan.

# Primera forma de estandarización.


compuestos["pmol"] = compuestos.pmol.apply(lambda x: sdz.standardize_mol(x))



# %%

# Segunda forma de estandarización.


# import chembl_structure_pipeline as csp

 
# compuestos["pmol"] = compuestos.pmol.apply(lambda x: csp.standardize_mol(x))


# %%

# forma compuesta

# compuestos["pmol"] = compuestos.inchi.apply(lambda x: sdz.standardize_mol(sdz.get_parent_mol(Chem.MolFromInchi(x))[0]))


# %%

# Se quitan los que no se hayan convertido de manera correcta.

compuestos = compuestos[pd.notna(compuestos.pmol)].reset_index(drop = True)


# %%

# Se busca que no queden moléculas compuestas.

from rdkit import Chem

compuestos["smiles"] = compuestos["pmol"].apply(lambda mol: Chem.MolToSmiles(mol))

mol_compuestas = compuestos[compuestos["smiles"].str.contains(r"\.")]

if len(mol_compuestas) > 0:
    print(mol_compuestas)
else:
    print("0")





# %%

from fgs import get_dec_fgs

# Se generan columnas donde se guardan la imagen con los grupos funcionales, lista de grupos funcionales, 
# SMILES de los grupos funcionales y moléculas de los grupos funcionales.

compuestos[["img_text","fgs","fgsmi","fgmol"]] = compuestos.apply(lambda x: pd.Series(get_dec_fgs(x["pmol"])), axis = 1)


# %%
 
# Se agrupan los grupos funcionales para ver cuantas veces aparecen en las moléculas.

#fg_dist_GONZALO = compuestos.explode(["fgsmi","fgmol"]).groupby("fgsmi", as_index = False).apply(lambda x: 
 #                                                                              pd.DataFrame({"fgsmi": [x["fgsmi"].iloc[0]], 
  #                                                                                           "n": [x.shape[0]], 
   #                                                                                        "mol": [x["fgmol"].iloc[0]]})).sort_values("n", ascending = False)



# %%

# Se agrupan los grupos funcionales para ver cuantas veces aparecen en las moléculas.

import pandas as pd

# Se crea un dataframe solo con lo que nos interesa, que es el tipo
# de compuesto (separados todos en líneas para poder contarlos), sus grupos funcionales
# y los objeto mol de éstos.

fg_dist_sep = compuestos.explode(["fgsmi","fgmol"])[["cmps", "fgsmi", "fgmol"]]


# Se cuentan los grupos funcionales por compuestos.

fg_dist_grupos = (fg_dist_sep.groupby(["cmps", "fgsmi"], as_index=False)
                  .agg(n=('fgsmi', 'size'), fgmol=('fgmol', 'first'))
                  .sort_values(["cmps", "n"], ascending=[True, False]))


#fg_dist_grupos = (fg_dist_sep.groupby(["cmps", "fgsmi"], as_index=False)
 #                 .agg(n=('fgsmi', 'size')),
  #                .sort_values(["cmps", "n"], ascending=[True, False]))
  
  
  
  
# CON VALUE COUNTS SALE EXACTAMENTE LO MISMO


# Usar value_counts para contar la frecuencia de "fgsmi" por "cmps"
# fg_dist_grupos = (fg_dist_sep.groupby("cmps")["fgsmi"]
#                  .value_counts()
 #                 .reset_index(name="n"))

# fg_dist_grupos = fg_dist_grupos.sort_values(["cmps", "n"], ascending=[True, False])


# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visualización de los grupos funcionales más comunes en los distintos compuestos
# mediante gráficos barplot.

top_20_fg = fg_dist_grupos.groupby('cmps').head(20)


for tipo in set(top_20_fg['cmps']):
    datos_tipo = top_20_fg[top_20_fg['cmps'] == tipo]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=datos_tipo, x='n', y='fgsmi', palette="Spectral")
    
    plt.title(f'Frecuencia de grupos funcionales en {tipo}')
    plt.xlabel('Frecuencia (n)')
    plt.ylabel('Grupos funcionales (fgsmi)')


# %%

from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display
import fgs

for tipo in set(top_20_fg['cmps']):
    datos_tipo = top_20_fg[top_20_fg['cmps'] == tipo]
    
    mols_tipo = datos_tipo['fgmol'].tolist()
    leyenda_tipo = datos_tipo['fgsmi'].tolist()
    
    imagen_moleculas = Draw.MolsToGridImage(mols_tipo, molsPerRow=5, subImgSize=(200, 200), legends=leyenda_tipo)

    display(imagen_moleculas)



# %%

import pandas as pd
import ast

# Convertir a lista en vez de cadenas de texto para poder contar.

compuestos['fgsmi'] = compuestos['fgsmi'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)



resultado_tabla20 = []

for tipo in compuestos['cmps'].unique():
    subset_conteo = compuestos[compuestos['cmps'] == tipo]
    
    n_moleculas = len(subset_conteo)
    
    # FG total (total count of FG in all the molecules of the set)
    
    subset_conteo['fg_count'] = subset_conteo['fgsmi'].apply(lambda fgsmi: len(fgsmi) if isinstance(fgsmi, list) else 0)
    total_fg = subset_conteo['fg_count'].sum()
    
    # FG total / mol (the previous count normalized by the number of molecules in the set)
    
    fg_total_mol = total_fg / n_moleculas
    
    # FG bin (binary count of FG in all the molecules in the set, i.e. only considering presence / absence of each FG)
    
    fg_bin = len(subset_conteo[subset_conteo['fg_count'] > 0])
    
    # FG bin / mol (the previous count normalized by the number of molecules in the set)
    
    fg_bin_mol = fg_bin / n_moleculas
    
    # FG un (count of unique FGs in the set)
    
    fg_unicos = len(subset_conteo['fgsmi'].explode().unique())
    
    # FG un / mol (the previous count normalized by the number of molecules in the set)
    
    fg_un_mol = fg_unicos / n_moleculas
    
    # Mols w/o FG (percentage of molecules lacking any FG)
    
    mols_sin_fg = len(subset_conteo[subset_conteo['fg_count'] == 0])
    porcentaje_mol_sin_fg = (mols_sin_fg / n_moleculas) * 100
    
    resultado_tabla20.append({
        'Compuesto': tipo,
        'n': n_moleculas,
        'FG (total)': total_fg,
        'FG (total) / mol': fg_total_mol,
        'FG (bin)': fg_bin,
        'FG (bin) / mol': fg_bin_mol,
        'FG (un)': fg_unicos,
        'FG (un) / mol': fg_un_mol,
        '% mols w/o FG': porcentaje_mol_sin_fg  # Añadimos el nuevo porcentaje
    })

tabla_distribucion_20 = pd.DataFrame(resultado_tabla20)


# %%

import pandas as pd

# Contar grupos funcionales aromáticos.

def contar_aromaticos(fgs):
    return sum(1 for fg in fgs if 'ar' in fg)

# Contar grupos funcionales con heteroátomos.

def contar_heteroatomos(fgs):
    return sum(1 for fg in fgs if any(atom in fg for atom in ['N', 'O','P', 'S', 'F', 'Cl', 'Br', 'I', 'At']))



resultado_tablaarhet = []


for tipo in compuestos['cmps'].unique():
    subset_conteo = compuestos[compuestos['cmps'] == tipo]
    
    n_moleculas = len(subset_conteo)
    
    # Ar FGs (bin) / mol (number of aromatic FGs (binary counts) by molecule)
    
    subset_conteo['ar_fg_count'] = subset_conteo['fgsmi'].apply(lambda fgs: contar_aromaticos(fgs) > 0)
    ar_fg_bin = subset_conteo['ar_fg_count'].sum()
    ar_fg_bin_mol = ar_fg_bin / n_moleculas
    
    # Ar FGs (un) (number of unique aromatic FGs)
    
    ar_fg_unicos = len(set(fg for fgs in subset_conteo['fgsmi'] for fg in fgs if 'ar' in fg))
    
    # Ar FGs (un) / mol (the previous count normalized by the number of molecules in the set)
    
    ar_fg_un_mol = ar_fg_unicos / n_moleculas
    
    #  Het FGs (bin) / mol (number of heteroatom FGs (binary counts) by molecule)
    
    subset_conteo['het_fg_count'] = subset_conteo['fgsmi'].apply(lambda fgs: contar_heteroatomos(fgs) > 0)
    het_fg_bin = subset_conteo['het_fg_count'].sum()
    het_fg_bin_mol = het_fg_bin / n_moleculas
    
    #  Het FGs (un) (number of unique heteroatom FG)
    
    het_fg_unicos = len(set(fg for fgs in subset_conteo['fgsmi'] for fg in fgs if any(atom in fg for atom in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'At'])))
    het_fg_un_mol = het_fg_unicos / n_moleculas
    
    
    resultado_tablaarhet.append({
        'Compuesto': tipo,
        'n': n_moleculas,
        'Ar FGs (bin)': ar_fg_bin,
        'Ar FGs (bin) / mol': ar_fg_bin_mol,
        'Ar FGs (un)': ar_fg_unicos,
        'Ar FGs (un) / mol': ar_fg_un_mol,
        'Het FGs (bin)': het_fg_bin,
        'Het FGs (bin) / mol': het_fg_bin_mol,
        'Het FGs (un)': het_fg_unicos,
        'Het FGs (un) / mol': het_fg_un_mol
    })

tabla_ar_het = pd.DataFrame(resultado_tablaarhet)



# %%

import pandas as pd
import numpy as np

atomos_objetivo = {'O': ['O'],'N': ['N'],'S': ['S'],'P': ['P'],'X': ['F', 'Cl', 'Br', 'I', 'At']}

# Contar los FGs binarios y únicos para cada átomo.

def contar_fgs_por_atomo(fgs, atomos):
    bin_count = sum(1 for fg in fgs if any(atom in fg for atom in atomos))
    unicos_count = len(set(fg for fg in fgs if any(atom in fg for atom in atomos)))
    return bin_count, unicos_count


resultados_tablaONSP = []


for tipo in compuestos['cmps'].unique():
    subset_conteo = compuestos[compuestos['cmps'] == tipo]
    
    n_moleculas = len(subset_conteo)
    
    resultado_compuesto = {'Compuesto': tipo, 'n': n_moleculas}
    
    for atomo, simbolos in atomos_objetivo.items():
        
        fg_bin_total = 0
        fg_un_total = 0
        
        # Contar los átomos que nos interesan en los GF
        
        for fgs in subset_conteo['fgsmi']:
            bin_count, unicos_count = contar_fgs_por_atomo(fgs, simbolos)
            fg_bin_total += bin_count
            fg_un_total += unicos_count
        
        
        fg_bin_por_mol = fg_bin_total / n_moleculas
        fg_un_por_mol = fg_un_total / n_moleculas
        
        
        resultado_compuesto[f'{atomo} FGs (bin) / mol'] = float(fg_bin_por_mol)
        resultado_compuesto[f'Frac {atomo} FGs (un)'] = float(fg_un_por_mol)
    

    resultados_tablaONSP.append(resultado_compuesto)


tablaONSP = pd.DataFrame(resultados_tablaONSP)


# %%

import pandas as pd

ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

tabla_distribucion_20.to_excel(f'{ruta_tablas}\\tabla_distribucion_20.xlsx', index=False)
tabla_ar_het.to_excel(f'{ruta_tablas}\\tabla_ar_het.xlsx', index=False)
tablaONSP.to_excel(f'{ruta_tablas}\\tablaONSP.xlsx', index=False)


# %%

import numpy as np
import pandas as pd


matrices = {}


for tipo in top_20_fg['cmps'].unique():
    top_fgs = top_20_fg[top_20_fg['cmps'] == tipo]['fgsmi'].tolist()
    subset_conteo = compuestos[compuestos['cmps'] == tipo]
    
    matriz_coocurrencia = np.zeros((20, 20), dtype=int)
    
    # Asignación indices a los FG para la matriz.
    
    fg_ind = {fg: i for i, fg in enumerate(top_fgs)}
    
    # Calcular la co-ocurrencia.
    
    for fgs in subset_conteo['fgsmi']:
        
        # FGs que están dentro del top 20.
        
        fgs_filt = [fg for fg in fgs if fg in top_fgs]
        
        # Matriz.
        
        for i in range(len(fgs_filt)):
            for j in range(i, len(fgs_filt)):
                index_i = fg_ind[fgs_filt[i]]
                index_j = fg_ind[fgs_filt[j]]
                matriz_coocurrencia[index_i, index_j] += 1
                if i != j:
                    matriz_coocurrencia[index_j, index_i] += 1
    
    matrices[tipo] = pd.DataFrame(matriz_coocurrencia, index=top_fgs, columns=top_fgs)


# %%

import seaborn as sns
import matplotlib.pyplot as plt


for tipo, matriz in matrices.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz, cmap='BuPu')
    plt.title(f'Matriz Co-ocurrencia {tipo}')
    plt.show()



