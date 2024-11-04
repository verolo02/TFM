# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:02:39 2024

@author: Vero
"""
# %%

# Python 3.12 30/10/24

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

fg_dist_sep = compuestos.explode(["fgsmi","fgmol"])[["cmps", "fgsmi", "fgmol", "foodb_id"]]


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

# IMAGEN DE LOS 20 GRUPOS FUNCIONALES MÁS COMUNES

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

# Count statistics of FGs in the different compound sets

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

# Count statistics of aromatic atom- and heteroatom-containing FGs in the different 
# compound sets

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

#  Count statistics of FGs containing O, N, S, P, or halogen atoms (X) by compound set

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

# EXPORTAR TABLAS A EXCEL

import pandas as pd

ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

tabla_distribucion_20.to_excel(f'{ruta_tablas}\\tabla_distribucion_20.xlsx', index=False)
tabla_ar_het.to_excel(f'{ruta_tablas}\\tabla_ar_het.xlsx', index=False)
tablaONSP.to_excel(f'{ruta_tablas}\\tablaONSP.xlsx', index=False)


# %%

# CREACIÓN MATRICES CORRELACIÓN

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
                    
    # Omitimos las diagonales.
    
    matriz_coocurrencia = matriz_coocurrencia.astype(float)
    np.fill_diagonal(matriz_coocurrencia, np.nan)
    
    matrices[tipo] = pd.DataFrame(matriz_coocurrencia, index=top_fgs, columns=top_fgs)


# %%

# REPRESENTACIÓN MATRIZ CORRELACIÓN

import seaborn as sns
import matplotlib.pyplot as plt


for tipo, matriz in matrices.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz, cmap='YlGnBu')
    plt.title(f'Matriz Co-ocurrencia {tipo}')
    plt.show()


# %%

# Contamos lás moléculas más comunes por grupo de compuesto.

# O QUERÍA LA MOLÉCULA QUE CONTENÍA EL GRUPO FUNCIONAL MÁS COMÚN? NO VEO SENTIDO.
# ES DECIR, LA MOLECULA DE LA QUE PROVIENE EL GRUPO FUNCIONAL
# O LAS PROPIEDADES DE TODAS LAS MOLÉCULAS QUE CONTENGAN ESE GRUPO FUNCIONAL??


top_smiles = (compuestos.groupby(["cmps", "smiles"], as_index=False)
              .agg(n=('smiles', 'size'), pmol=('pmol', 'first'))
              .sort_values(["cmps", "n"], ascending=[True, False]))

top_15_mol_prop = top_smiles.groupby("cmps").head(15)




# %%

# CÁLCULO PROPIEDADES 15 MOLÉCULAS MÁS COMUNES POR COMPUESTOS

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen
from rdkit.Chem import QED


# Sacamos las 15 moléculas más importantes por grupo de compuesto.

top_15_mol_prop = top_smiles.groupby("cmps").head(15)


# Función para sanitacizamos la mólecula, la preparamos para que no tenga errores a la hora de calcular las propiedades.

def preparar_molecula(mol):
    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None
    

# Función de cálculo de las propiedades.

def calcular_propiedades(mol):
    if mol is None:
        return None

    # Topological Polar Surface Area (TPSA).
    
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # LogP (octanol/water partition coefficient).
    
    logp = Crippen.MolLogP(mol)

    # Number of rotatable bonds (RB).
    
    rb = Descriptors.NumRotatableBonds(mol)

    # Number of hydrogen bond donors (HBD).
    
    hbd = rdMolDescriptors.CalcNumHBD(mol)

    # Number of hydrogen bond acceptors (HBA).
    
    hba = rdMolDescriptors.CalcNumHBA(mol)

    # Molecular weight (MW).
    
    mw = Descriptors.MolWt(mol)

    # Number of rings (nRing).
    
    nring = Descriptors.RingCount(mol)

    # Number of aromatic rings (nAromaticRing).
    
    naring = rdMolDescriptors.CalcNumAromaticRings(mol)
    
    # Quantitative Estimation of Drug-likeness (QED)
    
    qed_value = QED.qed(mol)  
    
    # Fracción de carbonos sp3 (Fsp3)
    
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    
    
    return [tpsa, logp, rb, hbd, hba, mw, nring, naring, qed_value, fsp3]


# Función para aplicar el cálculo de propiedades por compuesto.

def procesar_compuestos(row):
    mol = preparar_molecula(row['pmol'])
    propiedades = calcular_propiedades(mol)

    return pd.Series({
        'TPSA': propiedades[0],
        'LogP': propiedades[1],
        'Rotatable Bonds': propiedades[2],
        'HBD': propiedades[3],
        'HBA': propiedades[4],
        'MW': propiedades[5],
        'Number of Rings': propiedades[6],
        'Aromatic Rings': propiedades[7],
        'QED': propiedades[8],
        'Fsp3': propiedades[9]
    })

top_15_propiedades = top_15_mol_prop.apply(procesar_compuestos, axis=1)


top_15_propiedades['cmps'] = top_15_mol_prop['cmps'].values

# %%

# EXPORTAR TABLAS EXCEL

ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

top_15_propiedades.to_excel(f'{ruta_tablas}\\top_15_propiedades.xlsx', index=False)



# %%

# BOXPLOT CON OUTLIERS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Se realizan gráficos boxplots de las propiedades

# tpsa, logp, rb, hbd, hba, mw, nring, naring, qed_value, fsp3

propiedades_graficos = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings','Aromatic Rings', 'QED', 'Fsp3']

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10)) 
axes = axes.flatten() 

for i, propiedad in enumerate(propiedades_graficos):
    sns.boxplot(x='cmps', y=propiedad, data=top_15_propiedades, palette='hls', ax=axes[i])
    
    sns.swarmplot(x='cmps', y=propiedad, data=top_15_propiedades, color='black', ax=axes[i])

    axes[i].set_title(f'Distribución de {propiedad}', fontsize=16)
    axes[i].set_xlabel('Compuestos', fontsize=12)
    axes[i].set_ylabel(propiedad, fontsize=12)
    
plt.tight_layout()

# %%

# BOXPLOT SIN OUTLIERS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Filtramos outliers

def filtrar_outliers(outliers, propiedad):
    q1 = outliers[propiedad].quantile(0.25)
    q3 = outliers[propiedad].quantile(0.75)
    iqr = q3 - q1
    limites_inferiores = q1 - 1.5 * iqr
    limites_superiores = q3 + 1.5 * iqr
    return outliers[(outliers[propiedad] >= limites_inferiores) & (outliers[propiedad] <= limites_superiores)]


propiedades_graficos = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings','Aromatic Rings', 'QED', 'Fsp3']

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10)) 
axes = axes.flatten()  

for i, propiedad in enumerate(propiedades_graficos):
    top_15_filtrado = filtrar_outliers(top_15_propiedades, propiedad)

    sns.boxplot(x='cmps', y=propiedad, data=top_15_filtrado, palette='hls', ax=axes[i])
    sns.swarmplot(x='cmps', y=propiedad, data=top_15_filtrado, color='black', ax=axes[i])

    axes[i].set_title(f'Distribución de {propiedad}', fontsize=16)
    axes[i].set_xlabel('Compuestos', fontsize=12)
    axes[i].set_ylabel(propiedad, fontsize=12)

plt.tight_layout()
plt.show()


# %%

# VIOLINPLOT CON OUTLIERS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


propiedades_graficos = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings','Aromatic Rings', 'QED', 'Fsp3']

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10)) 
axes = axes.flatten() 

for i, propiedad in enumerate(propiedades_graficos):
    sns.violinplot(x='cmps', y=propiedad, data=top_15_propiedades, palette='hls', ax=axes[i])
    sns.swarmplot(x='cmps', y=propiedad, data=top_15_propiedades, color='black', ax=axes[i])

    axes[i].set_title(f'Distribución de {propiedad}', fontsize=16)
    axes[i].set_xlabel('Compuestos', fontsize=12)
    axes[i].set_ylabel(propiedad, fontsize=12)
    
plt.tight_layout()
plt.show()

# %%


# VIOLIN PLOTS SIN OUTLIERS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


propiedades_graficos = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings','Aromatic Rings', 'QED', 'Fsp3']

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10)) 
axes = axes.flatten() 

for i, propiedad in enumerate(propiedades_graficos):
    top_15_filtrado = filtrar_outliers(top_15_propiedades, propiedad)

    sns.violinplot(x='cmps', y=propiedad, data=top_15_filtrado, palette='hls', ax=axes[i], inner=None)
    sns.swarmplot(x='cmps', y=propiedad, data=top_15_filtrado, color='black', ax=axes[i])

    axes[i].set_title(f'Distribución de {propiedad}', fontsize=16)
    axes[i].set_xlabel('Compuestos', fontsize=12)
    axes[i].set_ylabel(propiedad, fontsize=12)

plt.tight_layout()
plt.show()


# %%

import pandas as pd
from scipy.stats import kruskal

# Realizamos la prueba de Krustal-Wallis sobre las propiedades de los distintos compuestos.

kruskal_resultados = {}

for propiedad in propiedades_graficos:
    grupos = [group[propiedad] for tipo, group in top_15_propiedades.groupby('cmps')]
  
    estadistico, p_valor = kruskal(*grupos)

    kruskal_resultados[propiedad] = {'estadistico': estadistico, 'p_valor': p_valor}

kruskal_resultados_df = pd.DataFrame(kruskal_resultados).T


# %%

ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

kruskal_resultados_df.to_excel(f'{ruta_tablas}\\kruskal_resultados_df.xlsx', index=True)

# %%

import scikit_posthocs as sp

# Realizamos la prueba de Conover sobre las propiedades que presentaban diferencias
# significativas.  BONFERRONI O HOLM??

conover_resultados = {}

for propiedad, resultado in kruskal_resultados.items():
    if resultado['p_valor'] < 0.05:
        posthoc_conover = sp.posthoc_conover(top_15_propiedades, val_col=propiedad, group_col='cmps', p_adjust='holm')
        conover_resultados[propiedad] = posthoc_conover
        
conover_resultados_df = pd.concat(conover_resultados, axis=0)


# %%

ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

conover_resultados_df.to_excel(f'{ruta_tablas}\\conover_resultados_df.xlsx', index=True)

# %%

import pandas as pd
import numpy as np
from itertools import combinations

# Calcular CLES entre dos grupos.

def calcular_cles(grupo1, grupo2):
    n = len(grupo1) * len(grupo2)
    conteo = sum(x > y for x in grupo1 for y in grupo2)
    return conteo / n


cles_resultados = {}

# Calculo de CLES para cada propiedad y par de grupos.

for propiedad in propiedades_graficos:
    grupos = top_15_propiedades.groupby('cmps')[propiedad].apply(list)
    pares_grupos = combinations(grupos.index, 2)
    
    cles_resultados[propiedad] = []
    
    for g1, g2 in pares_grupos:
        cles_valor = calcular_cles(grupos[g1], grupos[g2])
        
        # Calcular CLES.
        
        cles_resultados[propiedad].append({
            'Propiedad': propiedad,
            'Grupo1': g1,
            'Grupo2': g2,
            'CLES': cles_valor
        })
        
        # Calcular valor inverso para la otra comparación.
        
        cles_resultados[propiedad].append({
            'Propiedad': propiedad,
            'Grupo1': g2,
            'Grupo2': g1,
            'CLES': 1 - cles_valor
        })


df_cles = pd.DataFrame([resultado for resultados in cles_resultados.values() for resultado in resultados])


# %%

ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

df_cles.to_excel(f'{ruta_tablas}\\df_cles.xlsx', index=True)

# %%

# Añadir las clases químicas a todas las moléculas.

import pandas as pd


clases_quimicas = pd.read_csv(r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\new_compset_mycl.csv', sep=';')


compuestos = compuestos.merge(clases_quimicas[['foodb_id', 'inchi', 'mycl']], on=['foodb_id', 'inchi'], how='left')

# %%

# Añadir las clases químicas DE SUS MOLÉCULAS a los 15 grupos funcionales más comunes por compuesto.

fg_dist_sep = compuestos.explode(["fgsmi","fgmol"])[["cmps", "fgsmi", "fgmol", "foodb_id"]]

fg_dist_grupos = (fg_dist_sep.groupby(["cmps", "fgsmi"], as_index=False)
                  .agg(n=('fgsmi', 'size'), 
                       fgmol=('fgmol', 'first'), 
                       foodb_id=('foodb_id', 'first'))  
                  .sort_values(["cmps", "n"], ascending=[True, False]))



fg_dist_grupos = fg_dist_grupos.merge(compuestos[['foodb_id', 'mycl']].drop_duplicates(),on='foodb_id', how='left')

top_15_fg = fg_dist_grupos.groupby('cmps').head(15)



# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Heatmap grupos funcionales vs clase química de la molécula por compound set.

frec_heatmap_fgvsmycl = (top_15_fg.groupby(['cmps', 'fgsmi', 'mycl']).size().unstack(fill_value=0))


for tipo, frec_compoundset in frec_heatmap_fgvsmycl.groupby(level=0):
    frec_compoundset = frec_compoundset.droplevel(0)
    plt.figure(figsize=(15, 10))
    sns.heatmap(frec_compoundset, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={'label': 'Frecuencia'})
    plt.title(f"Frecuencia de grupos funcionales por clase química para {tipo}")
    plt.xlabel("Clases Químicas (mycl)")
    plt.ylabel("Grupos Funcionales (fgsmi)")
    plt.tight_layout()
    plt.show()


# %%

# Añadimos las dianas.

import pandas as pd

dianas = pd.read_excel(r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\dianas.xlsx')

# %%

# Se crea un dataframe con los compuestos que tienen dianas.

compuestos_dianas = pd.merge(compuestos, dianas, on='foodb_id', how='inner')


# %%

import pandas as pd

# Se crea un dataframe con las moléculas+dianas que contienen alguno de los 15 grupos
# funcionales más comunes.

compuestos_dianas_exploded = compuestos_dianas.explode('fgsmi')



# %%


top_fgsmi_por_compuesto_target = (compuestos_dianas_exploded.groupby(['cmps', 'target','target_class', 'fgsmi'], as_index=False)
    .agg(n=('fgsmi', 'size'),foodb_ids=('foodb_id', lambda x: list(x)))
    .sort_values(['cmps', 'n'], ascending=[True, False]))






