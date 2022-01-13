#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A python3 script computing bond and valence angle force constants using 
the Seminario (projected hessian) method.

@authors: Szymon Szrajer, Zuzanna Wojdyla, Tomasz Borowski

Last update: 13.01.2022
"""
import sys

from pyseminario_aux import fchk_read_n_atoms, fchk_read_atoms, fchk_read_hessian
from pyseminario_aux import Bond, Angle, triple_to_2_bond_labels, read_section_from_input


### ---------------------------------------------------------------------- ###
### Seting the file names                                                  ###
inp_file_name = sys.argv[1]


### ---------------------------------------------------------------------- ###
### Setting files and for which bonds and angles k will be calculated      ###

input_f = open(inp_file_name, 'r')

chk_file_names = read_section_from_input(input_f, '%FILES', '%END')
bonds_lines = read_section_from_input(input_f, '%BONDS', '%END')
angles_lines = read_section_from_input(input_f, '%ANGLES', '%END')

input_f.close()

bonds_2_calc = [] # a list of 2-element tuples with atom numbers in ascending order
for i in range(len(bonds_lines)):
    str_list = bonds_lines[i].split()
    num_list = []
    for item in str_list:
        num_list.append(eval(item))
    num_list.sort()
    bonds_2_calc.append(tuple(num_list))

angles_2_calc = []
for i in range(len(angles_lines)):
    str_list = angles_lines[i].split()
    num_list = []
    for item in str_list:
        num_list.append(eval(item))
    a = num_list[0]
    b = num_list[1]
    c = num_list[2]
    if a < c:
        ordered_num_list = [a, b, c]
    else:
        ordered_num_list = [c, b, a]
    angles_2_calc.append(tuple(ordered_num_list))


atoms_2_calc = [] # list of intigers - atom numbers (1-based)
for item in bonds_2_calc + angles_2_calc:
    for num in item:
        atoms_2_calc.append(num)
atoms_2_calc = list(set(atoms_2_calc))       
atoms_2_calc.sort()

##############################################################################
# Main loop through the chk files
results = [] # a list of dictionaries, one per fchk file

for i in range(len(chk_file_names)):
    fchk_file = chk_file_names[i][:-1] # remove '\n'
    fchk_file = './' + fchk_file # assumes the files are in the current dir

    res_dict = {}
    res_dict['file name'] = chk_file_names[i][:-1]

### ---------------------------------------------------------------------- ###
### Reading from fchk file                                                 ###
    fchk = open(fchk_file, 'r')
    
    N_atoms= 0 # int, number of atoms in the system
    atom_list = [] # a list of atom objects
    hessians_3x3 = {} # a dic with 3x3 hessians with (at_1, at_2) as keys
    
    N_atoms = fchk_read_n_atoms(fchk)
    atom_list = fchk_read_atoms(fchk,atoms_2_calc) 
    
    at_pairs_list = bonds_2_calc.copy()
    for triple in angles_2_calc:
        ab_label, bc_label = triple_to_2_bond_labels(triple)
        for label in [ab_label, bc_label]:
            if label not in at_pairs_list:
                at_pairs_list.append(label)
    
    for pair in at_pairs_list:
        hessians_3x3[pair] = fchk_read_hessian(fchk,pair)
    
    fchk.close()
### ---------------------------------------------------------------------- ###
### arranging data into lists of Bond and Angle objects                    ###

    bond_list = [] # a list of bond objects (needed for bond and angle k calculations) 
    angle_list =[] # a list of angle objects
    
    at_nr_2_seq = {} # a dict with atom number as key and seq number (0-based) in atom_list as value
    for i in range(len(atom_list)):
        at = atom_list[i]
        at_nr_2_seq[ at.get_number() ] = i
    
    for pair in at_pairs_list:
        at1 = atom_list[ at_nr_2_seq[ pair[0] ] ]
        at2 = atom_list[ at_nr_2_seq[ pair[1] ] ]
        two_atoms = [at1, at2]
        hess = hessians_3x3[ pair ]
        bond = Bond(two_atoms, hess)
        bond.set_label()
        bond_list.append( bond )
    
    for triple in angles_2_calc:
        ab_label, bc_label = triple_to_2_bond_labels(triple)
        ab_bond = None
        bc_bond = None
        for bond in bond_list:
            bond_lab = bond.get_label()
            if bond_lab == ab_label:
                ab_bond = bond
            elif bond_lab == bc_label:
                bc_bond = bond
        two_bonds = [ab_bond, bc_bond]
        angle = Angle( two_bonds )
        angle_list.append( angle )  
    
### ---------------------------------------------------------------------- ###
### calculations of force constants                                        ###

# preparatory calculations for bonds and angles
    for bond in bond_list:
        bond.set_bond_length()
        bond.set_u_ab()
        bond.set_eigs()
    
    for angle in angle_list:
        angle.set_label()
        angle.set_deg_value()
        angle.set_u_n()
        angle.set_u_pa()
        angle.set_u_pc()
        angle.set_k()


# actual calculations of k-values and storing the results
    for bond in bond_list:
        if bond.get_label() in bonds_2_calc:
            bond.set_k()
            label = bond.get_label()
            length = bond.get_bond_length()
            k = bond.get_k()
            res_dict[label] = [length, k]
                        
    for angle in angle_list:
        angle.set_k()
        label = angle.get_label()
        ang = angle.get_deg_value()
        k = angle.get_k()
        res_dict[label] = [ang, k]
    
    results.append(res_dict)    
    
### ---------------------------------------------------------------------- ###
### outputing the results                                                  ###

results_label_wise = {}
# aranging the results label-wise
for label in bonds_2_calc:
    res_list = []
    for item in results:
        f_name = item['file name']
        r = item[label][0]
        k = item[label][1]
        res_list.append( [f_name, r, k] )
    results_label_wise[label] = res_list

    
for label in angles_2_calc:
    res_list = []
    for item in results:
        f_name = item['file name']
        ang = item[label][0]
        k = item[label][1]
        res_list.append( [f_name, ang, k] )
    results_label_wise[label] = res_list

# printing the results
for label in bonds_2_calc:
    print('-----------------------------------------------')
    print("Bond " + str(label))
    print("File name    " + "r[A] " + "k[kcal/mol*A^2]")
    res_list = results_label_wise[label]
    for res in res_list:
        print(res[0] +'\t'+ '{:7.3f}'.format(res[1]) +'\t'+ '{:7.3f}'.format(res[2]))

for label in angles_2_calc:
    print('-----------------------------------------------')
    print("Angle " + str(label))
    print("File name    " + "angle[deg] " + "k[kcal/mol*rad^2]")
    res_list = results_label_wise[label]
    for res in res_list:
        print(res[0] +'\t'+ '{:7.3f}'.format(res[1]) +'\t'+ '{:7.3f}'.format(res[2]))



