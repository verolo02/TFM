# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:14:26 2024

@author: Vero
"""
# %%

# Python 3.12 30/10/24

import fgs
import pandas as pd
import rdkit
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
from rdkit import Chem
from rdkit.Chem import inchi
from chembl_structure_pipeline import standardizer as sdz
from fgs import get_dec_fgs
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen
from rdkit.Chem import QED
from rdkit.Chem import Draw
from IPython.display import display
from itertools import combinations
from scipy.stats import kruskal



# %%

# BLOQUE 1. IMPORTACIÓN Y CREACIÓN DE MOLÉCULAS.


# Se importan los datos.

ruta_archivo = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\fichero compuestos.csv'

compuestos = pd.read_csv(ruta_archivo, sep=';')


# Se extrae la estructura “parent” o principal del complejo de moléculas.

compuestos["pmol"] = compuestos.inchi.apply(lambda x: sdz.get_parent_mol(Chem.MolFromInchi(x))[0])


# Una vez obtenidas las estructuras "parent" se estandarizan.

compuestos["pmol"] = compuestos.pmol.apply(lambda x: sdz.standardize_mol(x))


# Se eliminan los compuestos no válidos.

compuestos = compuestos[pd.notna(compuestos.pmol)].reset_index(drop = True)


# Se busca que no queden moléculas compuestas para evitar errores en la posterior generación.

compuestos["smiles"] = compuestos["pmol"].apply(lambda mol: Chem.MolToSmiles(mol))

mol_compuestas = compuestos[compuestos["smiles"].str.contains(r"\.")]

if len(mol_compuestas) > 0:
    print(mol_compuestas)
else:
    print("0")


# Se generan columnas donde se guardan la imagen con los grupos funcionales, lista de grupos funcionales, 
# SMILES de los grupos funcionales y moléculas de los grupos funcionales.

compuestos[["img_text","fgs","fgsmi","fgmol"]] = compuestos.apply(lambda x: pd.Series(get_dec_fgs(x["pmol"])), axis = 1)



# %%

# BLOQUE 2. VISUALIZACIÓN DIFERENCIAS ENTRE GRUPOS FUNCIONALES POR CMPS (GRÁFICOS/IMÁGENES/TABLAS).

# Se agrupan los grupos funcionales para ver cuantas veces aparecen en las moléculas.

# Se crea un dataframe solo con lo que nos interesa, que es el tipo
# de compuesto (separados todos en líneas para poder contarlos), sus grupos funcionales
# y los objeto mol de éstos.

fg_dist_sep = compuestos.explode(["fgsmi","fgmol"])[["cmps", "fgsmi", "fgmol", "foodb_id", "pmol"]]


# Se cuentan los grupos funcionales por compuestos.

fg_dist_grupos = (fg_dist_sep.groupby(["cmps", "fgsmi"], as_index=False)
                  .agg(n=('fgsmi', 'size'), fgmol=('fgmol', 'first'))
                  .sort_values(["cmps", "n"], ascending=[True, False]))


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


# IMAGEN DE LOS 20 GRUPOS FUNCIONALES MÁS COMUNES POR CMPS.


for tipo in set(top_20_fg['cmps']):
    datos_tipo = top_20_fg[top_20_fg['cmps'] == tipo]
    
    mols_tipo = datos_tipo['fgmol'].tolist()
    leyenda_tipo = datos_tipo['fgsmi'].tolist()
    
    imagen_moleculas = Draw.MolsToGridImage(mols_tipo, molsPerRow=5, subImgSize=(200, 200), legends=leyenda_tipo)

    display(imagen_moleculas)



# Count statistics of FGs in the different compound sets.


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
    
    subset_conteo['fg_bin_count'] = subset_conteo['fgsmi'].apply(lambda fgsmi: len(set(fgsmi)) if isinstance(fgsmi, list) else 0)
    fg_bin = subset_conteo['fg_bin_count'].sum()
    
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
        '% mols w/o FG': porcentaje_mol_sin_fg
    })

tabla_distribucion_20 = pd.DataFrame(resultado_tabla20)



# Count statistics of aromatic atom- and heteroatom-containing FGs in the different 
# compound sets.


resultado_tablaarhet = []

def es_aromatico(fg):
    return 'ar' in fg


def es_heteroatomo(fg):
    return any(atom in fg for atom in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'At'])


for tipo in compuestos['cmps'].unique():
    subset_conteo = compuestos[compuestos['cmps'] == tipo]
    
    n_moleculas = len(subset_conteo)
    
    # Ar FGs (bin) / mol (number of aromatic FGs (binary counts) by molecule)

    subset_conteo['ar_fg_bin_count'] = subset_conteo['fgsmi'].apply(lambda fgs: sum(1 for fg in fgs if es_aromatico(fg)) if isinstance(fgs, list) else 0)
    ar_fg_bin = subset_conteo['ar_fg_bin_count'].sum()
    ar_fg_bin_mol = ar_fg_bin / n_moleculas 
    
    # Ar FGs (un) (number of unique aromatic FGs)
    
    ar_fg_unicos = len(set(fg for fgs in subset_conteo['fgsmi'] for fg in fgs if es_aromatico(fg)))
    
    # Ar FGs (un) / mol (the previous count normalized by the number of molecules in the set)
    
    ar_fg_un_mol = ar_fg_unicos / n_moleculas
    
    #  Het FGs (bin) / mol (number of heteroatom FGs (binary counts) by molecule)
    
    subset_conteo['het_fg_bin_count'] = subset_conteo['fgsmi'].apply(lambda fgs: sum(1 for fg in fgs if es_heteroatomo(fg)) if isinstance(fgs, list) else 0)
    het_fg_bin = subset_conteo['het_fg_bin_count'].sum()
    het_fg_bin_mol = het_fg_bin / n_moleculas  # Normalizado por el número de moléculas
    
    #  Het FGs (un) (number of unique heteroatom FG)
    
    het_fg_unicos = len(set(fg for fgs in subset_conteo['fgsmi'] for fg in fgs if es_heteroatomo(fg)))
   
    # Het FGs (un) / mol (the previous count normalized by the number of molecules in the set)
   
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



#  Count statistics of FGs containing O, N, S, P, or halogen atoms (X) by compound set


atomos_objetivo = {'O': ['O'],'N': ['N'],'S': ['S'],'P': ['P'],'X': ['F', 'Cl', 'Br', 'I', 'At']}

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

        for fgs in subset_conteo['fgsmi']:
            bin_count, unicos_count = contar_fgs_por_atomo(fgs, simbolos)
            
            #  CUenta todas las ocurrencias de cada tipo de fgsmi que contenga el átomo de interés.
            
            fg_bin_total += bin_count
            
            # Si al menos un tipo de grupo funcional que contiene el átomo de interés está presente, cuenta 1 por molécula.
            
            fg_un_total += (1 if unicos_count > 0 else 0)
            
        fg_bin_por_mol = fg_bin_total / n_moleculas
        fg_un_por_mol = fg_un_total / n_moleculas
        
        resultado_compuesto[f'{atomo} FGs (bin) / mol'] = float(fg_bin_por_mol)
        resultado_compuesto[f'Frac {atomo} FGs (un)'] = float(fg_un_por_mol)
    
    resultados_tablaONSP.append(resultado_compuesto)


tablaONSP = pd.DataFrame(resultados_tablaONSP)



# EXPORTAR TABLAS A EXCEL


ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

tabla_distribucion_20.to_excel(f'{ruta_tablas}\\tabla_distribucion_20.xlsx', index=False)
tabla_ar_het.to_excel(f'{ruta_tablas}\\tabla_ar_het.xlsx', index=False)
tablaONSP.to_excel(f'{ruta_tablas}\\tablaONSP.xlsx', index=False)


# %%

# BLOQUE 3. DIFERENCIAS POR PARES DE GRUPOS POR CMPS.


# CREACIÓN MATRICES CORRELACIÓN


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



# REPRESENTACIÓN MATRIZ CORRELACIÓN


for tipo, matriz in matrices.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz, cmap='YlGnBu')
    plt.title(f'Matriz Co-ocurrencia {tipo}')
    plt.show()



# %%

# BLOQUE 4. OBTENCIÓN PROPIEDADES FISICOQUÍMICAS. REPRESENTACIÓN GRÁFICOS.


# Obtengo los 15 grupos funcionales más comunes por cmps, y despúes los FG comunes 
# entre los 3 compuestos (dentro de esos 15). Le añado la posición que ocupan 
# dentro de su cmps para información.


def obtener_top_15_fgsmi_comunes(fg_dist_sep):
    top_15_fgsmi = (fg_dist_sep.groupby(['cmps', 'fgsmi']).size().reset_index(name='n')
                    .sort_values(['cmps', 'n'], ascending=[True, False]).groupby('cmps').head(15))
    
    resultado = (fg_dist_sep.merge(top_15_fgsmi, on=['cmps', 'fgsmi'], how='inner')
                 .groupby(['cmps', 'fgsmi']).agg(pmol=('pmol', list), n=('n', 'first'))
                 .reset_index().sort_values(['cmps', 'n'], ascending=[True, False]))

    resultado['posicion_fgsmi'] = resultado.groupby('cmps').cumcount() + 1
    
    grupos_funcionales_comunes = set.intersection(*resultado.groupby('cmps')['fgsmi'].apply(set))
    
    resultado_comunes = resultado[resultado['fgsmi'].isin(grupos_funcionales_comunes)]

    return resultado_comunes

top_15_prop4_comunes = obtener_top_15_fgsmi_comunes(fg_dist_sep)



# EXPORTAR A EXCEL. (No se abre bien en el panel ya que es muy grande el dataframe).

ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

top_15_prop4_comunes.to_excel(f'{ruta_tablas}\\top_15_prop4_comunes.xlsx', index=True)



# Para que sea más manejable para la función de cálculo de propiedades hago explode
# y le añado el foodb_id para información.

top_15_propiedades_expandidas4 = (top_15_prop4_comunes.explode('pmol').reset_index(drop=True)
                                  .merge(compuestos[['pmol', 'foodb_id']], on='pmol', how='left'))



# Almaceno las moléculas por cada FG común (mejor visualización de los resultados).

dataframes_por_fgsmi = {}

for fgsmi, grupo in top_15_propiedades_expandidas4.groupby('fgsmi'):
    dataframes_por_fgsmi[fgsmi] = grupo.reset_index(drop=True)



# FUNCIÓN CÁLCULO PROPIEDADES MOLÉCULAS


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



# Calculo las propiedades y le quito las moléculas parent duplicadas.

for fgsmi, df in dataframes_por_fgsmi.items():
    df_sin_duplicados = df.drop_duplicates(subset=['pmol']).reset_index(drop=True)
    propiedades_df = df_sin_duplicados.apply(procesar_compuestos, axis=1)
    dataframes_por_fgsmi[fgsmi] = pd.concat([df_sin_duplicados, propiedades_df], axis=1)
    
    

# BOXPLOT SIN OUTLIERS PROPIEDADES por FG común.


propiedades = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings', 'Aromatic Rings', 'QED', 'Fsp3']


def filtrar_outliers(df, propiedad):
    def filtrar_grupo(grupo):
        q1 = grupo[propiedad].quantile(0.25)
        q3 = grupo[propiedad].quantile(0.75)
        iqr = q3 - q1
        limites_inferiores = q1 - 1.5 * iqr
        limites_superiores = q3 + 1.5 * iqr
        return grupo[(grupo[propiedad] >= limites_inferiores) & (grupo[propiedad] <= limites_superiores)]
    
    return df.groupby('cmps').apply(filtrar_grupo).reset_index(drop=True)


for fgsmi, df in dataframes_por_fgsmi.items():

    for propiedad in propiedades:
        df = filtrar_outliers(df, propiedad)

    plt.figure(figsize=(15, 10))
    

    for i, propiedad in enumerate(propiedades):
        plt.subplot(2, 5, i+1)

        sns.boxplot(x=df['cmps'], y=df[propiedad], palette='hls')

        plt.title(f'{propiedad}')
        plt.xticks(rotation=90)

    plt.tight_layout()
    plt.suptitle(f'Propiedades para {fgsmi}', fontsize=16, y=1.05)



# VIOLINPLOT SIN OUTLIERS PROPIEDADES por fg común.


propiedades = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings', 'Aromatic Rings', 'QED', 'Fsp3']


for fgsmi, df in dataframes_por_fgsmi.items():

    for propiedad in propiedades:
        df = filtrar_outliers(df, propiedad)

    plt.figure(figsize=(15, 10))
    

    for i, propiedad in enumerate(propiedades):
        plt.subplot(2, 5, i+1)

        sns.violinplot(x=df['cmps'], y=df[propiedad], palette='hls')

        plt.title(f'{propiedad}')
        plt.xticks(rotation=90)

    plt.tight_layout()
    plt.suptitle(f'Propiedades para {fgsmi}', fontsize=16, y=1.05)
    
    
# %%

# BLOQUE 5. ESTUDIOS ESTADÍSTICOS PROPIEDADES.


#Krustal-Wallis sobre las propiedades para CADA fgsmi COMÚN por cmps.


propiedades = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings', 'Aromatic Rings', 'QED', 'Fsp3']


kruskal_resultados_totales = {}


for fgsmi, df in dataframes_por_fgsmi.items():
    kruskal_resultados = {}  
    
    for propiedad in propiedades:
        grupos = [group[propiedad].dropna() for tipo, group in df.groupby('cmps')]
        
        estadistico, p_valor = kruskal(*grupos)
        
        kruskal_resultados[propiedad] = {'estadistico': estadistico, 'p_valor': p_valor}
    
    kruskal_resultados_totales[fgsmi] = pd.DataFrame(kruskal_resultados).T



# TABLAS EXCEL POR DATAFRAME.


with pd.ExcelWriter(r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel\kruskal_resultados_totales.xlsx', engine='openpyxl') as writer:
    for nombre_hoja, df in kruskal_resultados_totales.items():
        nombre_hoja_limpio = "".join(c for c in nombre_hoja if c.isalnum() or c in " _-")
        df.to_excel(writer, sheet_name=nombre_hoja_limpio, index=True)


# CONOVER SOBRE KRUSTAL-WALLIS (HOLM).

conover_resultados_totales = {}


for fgsmi, df in dataframes_por_fgsmi.items():
    conover_resultados = {}
    
    for propiedad, resultado in kruskal_resultados_totales[fgsmi].iterrows():
        if resultado['p_valor'] < 0.05:
            posthoc_conover = sp.posthoc_conover(df, val_col=propiedad, group_col='cmps', p_adjust='holm')
            conover_resultados[propiedad] = posthoc_conover
            
    if conover_resultados:
        conover_resultados_totales[fgsmi] = conover_resultados
        

# TABLAS EXCEL POR DATAFRAME.

with pd.ExcelWriter(r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel\conover_resultados_totales.xlsx', engine='openpyxl') as writer:
    for fgsmi, conover_resultados in conover_resultados_totales.items():
        for propiedad, conover_df in conover_resultados.items():
            nombre_hoja = f"{fgsmi}_{propiedad}"
            nombre_hoja_limpio = "".join(c for c in nombre_hoja if c.isalnum() or c in " _-")
            
            conover_df.to_excel(writer, sheet_name=nombre_hoja_limpio, index=True)



# CLES.

propiedades = ['TPSA', 'LogP', 'Rotatable Bonds', 'HBD', 'HBA', 'MW', 'Number of Rings', 'Aromatic Rings', 'QED', 'Fsp3']


def calcular_cles(grupo1, grupo2):
    n = len(grupo1) * len(grupo2)
    conteo = sum(x > y for x in grupo1 for y in grupo2)
    return conteo / n


cles_resultados_totales = []


for fgsmi, df in dataframes_por_fgsmi.items():
    for propiedad in propiedades:

        grupos = df.groupby('cmps')[propiedad].apply(list)

        pares_grupos = combinations(grupos.index, 2)
        
        for g1, g2 in pares_grupos:
            cles_valor = calcular_cles(grupos[g1], grupos[g2])
            
            cles_resultados_totales.append({
                'Grupo Funcional': fgsmi,
                'Propiedad': propiedad,
                'Grupo1': g1,
                'Grupo2': g2,
                'CLES': cles_valor
            })
            cles_resultados_totales.append({
                'Grupo Funcional': fgsmi,
                'Propiedad': propiedad,
                'Grupo1': g2,
                'Grupo2': g1,
                'CLES': 1 - cles_valor
            })


df_cles = pd.DataFrame(cles_resultados_totales)


# TABLAS EXCEL POR DATAFRAME.


ruta_tablas = r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\tablas excel'

df_cles.to_excel(f'{ruta_tablas}\\df_cles.xlsx', index=True)


# %%

# BLOQUE 6. FG VS CLASES QUÍMICAS.

# Añadir las clases químicas a todas las moléculas.


clases_quimicas = pd.read_csv(r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\new_compset_mycl.csv', sep=';')


compuestos = compuestos.merge(clases_quimicas[['foodb_id', 'inchi', 'mycl']], on=['foodb_id', 'inchi'], how='left')


# Obtenemos los 15 grupos funcionales más comunes por compuestos y agrupamos
# las moléculas que contengan cada grupo funcional.


def obtener_top_15_fgsmi(fg_dist_sep):
    return (fg_dist_sep.groupby(['cmps', 'fgsmi']).agg(pmol=('pmol', list), n=('fgsmi', 'size'))
        .reset_index().sort_values(['cmps', 'n'], ascending=[True, False])
        .groupby('cmps').head(15))


top_15_heatmap = obtener_top_15_fgsmi(fg_dist_sep)



# Crear el dataframe con todas las moléculas que contengan los primeros 15 FG
# por cmps y añadirles la clase química.

# QUITANDO DUPLICADOS, ya que nos interesa si la molécuña tiene o no tiene el FG.


top_15_heatexp = (top_15_heatmap.explode('pmol').reset_index(drop=True))

top_15_heatexp = top_15_heatexp.merge(compuestos[['pmol', 'mycl', 'foodb_id']],on='pmol',how='left')

top_15_heatexp = top_15_heatexp.drop_duplicates(subset=['cmps', 'fgsmi', 'mycl', 'pmol'])



frec_heatmap_fgvsmycl = (top_15_heatexp.groupby(['cmps', 'fgsmi', 'mycl']).size().unstack(fill_value=0))


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

# BLOQUE 7. FG, CMPS Y SUS DIANAS.


# Añadimos las dianas.

dianas = pd.read_excel(r'C:\Users\Vero\Desktop\UOC\3 cuatri\TFM\dianas.xlsx')


# Se crea un dataframe con los compuestos que tienen dianas.

compuestos_dianas = pd.merge(compuestos, dianas, on='foodb_id', how='inner')


# Se crea un dataframe con las moléculas+dianas.

compuestos_dianas_exploded = compuestos_dianas.explode('fgsmi')


# Se crea un dataframe para ver dentro de cada cmps que grupo funcional está asociado
# a qué dianas farmacológicas.

# Agrupamos por 'cmps', 'fgsmi', 'target' y 'target_class' y contamos las ocurrencias.

fgsmi_compuesto_diana = (compuestos_dianas_exploded.groupby(['cmps', 'fgsmi', 'target', 'target_class'], as_index=False)
    .agg(n=('target', 'size')))


# Se seleccionan las 5 'target_class' más frecuentes para cada combinación de 'cmps' y 'fgsmi'
# dentro de los 15 FG comunes.

top_5_target_class_per_cmps_fgsmi = (fgsmi_compuesto_diana[fgsmi_compuesto_diana['fgsmi'].isin(top_15_prop4_comunes['fgsmi'])]
    .sort_values(['cmps', 'fgsmi', 'n'], ascending=[True, True, False]).groupby(['cmps', 'fgsmi']).head(5))


# Gráfico de las target class más frecuentes por cmps y fgsmi.

ncols = 4
nrows = 2 

plt.figure(figsize=(18, 6 * nrows))


for i, fgsmi in enumerate(top_5_target_class_per_cmps_fgsmi['fgsmi'].unique()):
    plt.subplot(nrows, ncols, i + 1)
    
    fgsmi_data = top_5_target_class_per_cmps_fgsmi[top_5_target_class_per_cmps_fgsmi['fgsmi'] == fgsmi]

    sns.barplot(
        data=fgsmi_data,
        x='target_class',
        y='n',
        hue='cmps',
        dodge=True)
    
    plt.title(f'{fgsmi}', fontsize=14)
    plt.xlabel('Target Class', fontsize=12)
    plt.ylabel('Número de Ocurrencias (n)', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(fontsize='small')

plt.tight_layout()



