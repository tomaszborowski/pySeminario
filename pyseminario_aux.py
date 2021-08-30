#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a collection of classes and functions for pyseminario

Authors: Szymon Szrajer, Tomasz Borowski, Zuzanna Wojdyla

branch: zuza_debug
"""

import numpy as np
import re

Bohr_to_A = 0.52917721
Hartree_to_kcal = 627.509608
hess_au_to_kcal_A = Hartree_to_kcal / ( Bohr_to_A**2 ) 


# at_num_symbol - dic mapping atomic number to element symbol (up to 86 - Rn)
at_num_symbol = \
    {1: 'H', 2: 'He',
     3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
     11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
     19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
     30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
     37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag',
     48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
     55: 'Cs', 56: 'Ba', 57: 'La', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
     80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn',
     58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er',
     69: 'Tm', 70: 'Yb', 71: 'Lu'}


class Atom(object):
    atomic_number = 0
    symbol = ""
    number = 0
    coordinates = np.empty((1, 3), dtype=np.float64)  # coordinates in Bohrs
    angs_coordinates = np.empty((1, 3), dtype=np.float64)  # coordinates in Angstroms

    # the class "constructor"
    def __init__(self, atomic_number, coordinates):
        self.atomic_number = atomic_number
        self.symbol = at_num_symbol[atomic_number]
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.angs_coordinates = self.coordinates * Bohr_to_A

    def get_symbol(self):
        return self.symbol
    def get_number(self):
        return self.number
    def get_angs_coords(self):
        return self.angs_coordinates

    def set_number(self,nr):
        self.number = nr


def vector_ab(atom_1,atom_2):
    return (atom_2.get_angs_coords() - atom_1.get_angs_coords())


def distance_A_2_atoms(atom_1,atom_2):
    return np.linalg.norm( vector_ab(atom_1,atom_2) )


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


class Bond(object):
    label = None # tuple with 1-based atom numbers in ascending order
    atoms = []  # list of 2 Atom objects
    bond_length = 0.0 # bond length in A
    u_ab = None # unit length [A] vector from A to B
    hessian = None  # hessian in Hartrees/Bohr^2
    kcm_hessian = None  # hessian in kilocalories/(mol * A^2)
    kcm_hessian_T = None # transposed kcm_hessian
    lambdas_ab = None # eigenvalues for kcm_hessian
    eig_vecs_ab = None # eigenvectors for kcm_hessian
    lambdas_ba = None # eigenvalues for kcm_hessian_T
    eig_vecs_ba = None # eigenvectors for kcm_hessian_T
    k = None # force constant for the bond A-B [kcal/(mol A^2)]
    
    # the class "constructor"
    def __init__(self, atoms, hessian):
        self.atoms = atoms
        self.hessian = hessian
        self.kcm_hessian = self.hessian * hess_au_to_kcal_A
        self.kcm_hessian_T = np.transpose(self.kcm_hessian)

    def get_label(self):
        return self.label
    def get_atoms(self):
        return self.atoms
    def get_bond_length(self):
        return self.bond_length
    def get_u_ab(self):
        return self.u_ab
    def get_kcm_hessian(self):
        return self.kcm_hessian
    def get_kcm_hessian_T(self):
        return self.kcm_hessian_T
    def get_lambdas_ab(self):
        return self.lambdas_ab
    def get_eig_vecs_ab(self):
        return self.eig_vecs_ab
    def get_lambdas_ba(self):
        return self.lambdas_ba
    def get_eig_vecs_ba(self):
        return self.eig_vecs_ba
    def get_k(self):
        return self.k
    
    def set_label(self):
        temp = []
        for at in self.atoms:
            temp.append( at.get_number() )
        temp.sort()
        self.label = tuple(temp)
    def set_bond_length(self):
        atom_1 = self.atoms[0]
        atom_2 = self.atoms[1]
        self.bond_length = distance_A_2_atoms(atom_1,atom_2)
    def set_u_ab(self):
        atom_1 = self.atoms[0]
        atom_2 = self.atoms[1]
        self.u_ab = normalize_vector( vector_ab(atom_1,atom_2) )
    def set_eigs(self):
        self.lambdas_ab, self.eig_vecs_ab = np.linalg.eig(-1. * self.kcm_hessian)
        self.lambdas_ba, self.eig_vecs_ba = np.linalg.eig(-1. * self.kcm_hessian_T)
    def set_k(self):
        p1 = 0.5 * sum(self.lambdas_ab[i] * np.abs(np.dot(self.u_ab, self.eig_vecs_ab[:, i])) for i in range(3))
        p2 = 0.5 * sum(self.lambdas_ba[i] * np.abs(np.dot(self.u_ab, self.eig_vecs_ba[:, i])) for i in range(3))
        self.k = p1 + p2


class Angle(object):
    label = None # tuple with 1-based atom numbers 
    bonds = None # a list of 2 Bond objects
    u_n = np.empty((1, 3), dtype=np.float64)  # unit length vector normal to the A-B-C plane
    u_pa = np.empty((1, 3), dtype=np.float64)  # unit legth vector orthogonal to A-B bond
    u_pc = np.empty((1, 3), dtype=np.float64)  # unit legth vector orthogonal to B-C bond
    deg_value = None # value in degrees
    k = None # force constant for the angle A-B-C [kcal/(mol rad^2)]
    
    # the class "constructor"
    def __init__(self, bonds):
        self.bonds = bonds

    def get_label(self):
        return self.label
    def get_bonds(self):
        return self.bonds
    def get_deg_value(self):
        return self.deg_value
    def get_k(self):
        return self.k

    def set_label(self):
        at_nbrs = []
        for bd in self.bonds:
            for at in bd.get_atoms():
                at_nbrs.append( at.get_number() )
        for item in at_nbrs:
            if at_nbrs.count(item) == 2:
                b = item
                at_nbrs.remove(item)
                at_nbrs.remove(item)
                break
        at_nbrs.sort()                        
        self.label = tuple( [at_nbrs[0], b,  at_nbrs[1]] )
    def set_deg_value(self):   
        label_ab = self.bonds[0].get_label()
        label_cb = self.bonds[1].get_label()
        u_ab = self.bonds[0].get_u_ab()
        u_cb = self.bonds[1].get_u_ab()
        if label_ab[0] == label_cb[1] or label_ab[1] == label_cb[0]:
            u_cb = -1.0 * u_cb
        self.deg_value = np.degrees( np.arccos( np.dot(u_ab, u_cb) ) )
    def set_u_n(self):
        u_ab = self.bonds[0].get_u_ab()
        u_cb = self.bonds[1].get_u_ab()
        u_n = np.cross(u_cb, u_ab)
        self.u_n = normalize_vector( u_n )
    def set_u_pa(self):
        u_ab = self.bonds[0].get_u_ab()
        self.u_pa = np.cross(self.u_n, u_ab)        
    def set_u_pc(self):
        u_cb = self.bonds[1].get_u_ab()
        self.u_pc = np.cross(u_cb, self.u_n)
    def set_k(self):
        label_ab = self.bonds[0].get_label()
        label_cb = self.bonds[1].get_label()
        if label_ab[0] == label_cb[0]: # take care to take AB and CB eigenvalues/vectors for A-B-C
            lambdas_ab = self.bonds[0].get_lambdas_ba()
            lambdas_cb = self.bonds[1].get_lambdas_ba()        
            eig_vecs_ab = self.bonds[0].get_eig_vecs_ba()
            eig_vecs_cb = self.bonds[1].get_eig_vecs_ba()
        elif label_ab[1] == label_cb[0]:
            lambdas_ab = self.bonds[0].get_lambdas_ab()
            lambdas_cb = self.bonds[1].get_lambdas_ba()        
            eig_vecs_ab = self.bonds[0].get_eig_vecs_ab()
            eig_vecs_cb = self.bonds[1].get_eig_vecs_ba()
        elif label_ab[1] == label_cb[1]:
            lambdas_ab = self.bonds[0].get_lambdas_ab()
            lambdas_cb = self.bonds[1].get_lambdas_ab()        
            eig_vecs_ab = self.bonds[0].get_eig_vecs_ab()
            eig_vecs_cb = self.bonds[1].get_eig_vecs_ab()            
        
        r_ab = self.bonds[0].get_bond_length()
        r_cb = self.bonds[1].get_bond_length()
        #
        p12 = sum(lambdas_ab[i] * np.abs(np.dot(self.u_pa, eig_vecs_ab[:, i])) for i in range(3))
        p1 = r_ab**2 * p12 # no averaging, movement of A
        #
        p22 = sum(lambdas_cb[i] * np.abs(np.dot(self.u_pc, eig_vecs_cb[:, i])) for i in range(3))
        p2 = r_cb**2 * p22 # no averaging, movement of C
        
        k_inv = (1.0 / p1) + (1.0 / p2)
        self.k = 1.0 / k_inv
 
       
def fchk_read_n_atoms(file):
    """
    Reads number of atoms from the fchk file
    Parameters
    ----------
    file : fchk file (file object)
    Returns
    -------
    natom (int)
    """
    # w pliku file wyszukaj odp linii
    file.seek(0)
    flag_line = "Number of atoms"
    while True:
        try:
           a = file.readline()
        except UnicodeDecodeError:
           pass
        if not a:
            break
        if a[0] != ' ':
            match_flag=re.search(flag_line,a)
            if match_flag:
               natom = eval(a.split()[4]) 
    return natom


def which_line_pos(seq_number, n_per_line):
    """ calculates 1-based line numbers and positions in a line
    for data formated n_per_line items per line;
    seq_number : (int) 1-based number of the data in the sequence
    n_per_line : (int) number of fields/items per line"""
    div = seq_number // n_per_line
    rem = seq_number % n_per_line
    if rem != 0:
        which_position = rem
    else:
        which_position = n_per_line
    if rem != 0:
        which_line = div+1
    else:
        which_line = div
    return which_line, which_position


def gen_ster_list(line_pos_in_seq):
    """ generate a stearing list (used to read from 5-per-line block of data) from a list of 2-element tuples (line_nr, column_nr) 
    Input
    line_pos_in_seq : list of 2-element tuples, [(line_nr, column_nr), ...]
    Returns
    ster_list : a list of lists, each telling line number, how many elements to be read from a line and which elements
    """
    ster_list = [] 
    line_pos_in_seq_len = len(line_pos_in_seq)
    for i in range(line_pos_in_seq_len):
        ele = line_pos_in_seq[i]
        count = 0
        pos_list = []
        l_nr = ele[0]
        if l_nr not in [ster_list[i][0] for i in range(len(ster_list))]:
            for j in range(i,line_pos_in_seq_len,1):
                elemt = line_pos_in_seq[j]
                if elemt[0] == l_nr:
                    count += 1
                    pos_list.append(elemt[1])
            ster_list.append([l_nr, count, pos_list]) 
    return ster_list


def fchk_read_atoms(file, at_numbers):
    """
    Parameters
    ----------
    file : fchk file (file object)
    at_numbers : a list of atom numbers (1-based and sorted ascending) to be read

    Returns
    -------
    atom_list : a list of Atom objects
    """
    atomic_numbers = []
    atomic_coordinates = []
    atom_list = []
    # calculate line and position in line with the atomic numbers 
    which_line = [] # list of relative 1-based line numbers with at numbers for atoms
    which_position = [] # list of 1-based position mumbers with at numbers for atoms
    for at_nr in at_numbers:
        line, pos = which_line_pos(at_nr, 6)
        which_position.append(pos)
        which_line.append(line)
    max_line_nr = which_line[-1]
    n_atom_2_read = len(at_numbers)
    # w pliku file wyszukaj odp linii
    file.seek(0)
    flag_line = "Atomic numbers"
    while True:
        try:
           a = file.readline()
        except UnicodeDecodeError:
           pass
        if not a:
            break
        if a[0] != ' ':
            match_flag=re.search(flag_line,a)
            if match_flag:
               line_counter = 0
               atom_counter = 0
               for i in range(max_line_nr):
                   line_counter += 1
                   a = file.readline()
                   if line_counter == which_line[atom_counter]:
                       a_split = a.split()
                       l_a_split = len(a_split)
                       for j in range(l_a_split):
                           if j+1 == which_position[atom_counter]:
                               atomic_numbers.append( eval(a_split[j]) )
                               atom_counter += 1
                               if atom_counter >= n_atom_2_read:
                                   break   
    line_pos_in_seq = []
    for at_nr in at_numbers:
        for i in range(3):
            line, pos = which_line_pos(3*at_nr-2+i, 5)
            line_pos_in_seq.append( (line, pos) )
    skip_lines_tab = [] # a table with number of lines to be skipped   
    for i in range(3*n_atom_2_read):
        if i == 0:
            skip_lines_tab.append( line_pos_in_seq[i][0] - 1 )
        else:
            jump = ( line_pos_in_seq[i][0] - line_pos_in_seq[i-1][0] - 1 )
            if jump != -1:
                skip_lines_tab.append( jump )    
    ster_list = gen_ster_list(line_pos_in_seq)             
    # w pliku file wyszukaj odp linii
    file.seek(0)
    flag_line = "Current cartesian coordinates"
    while True:
        try:
           a = file.readline()
        except UnicodeDecodeError:
           pass
        if not a:
            break
        if a[0] != ' ':
            match_flag=re.search(flag_line,a)
            if match_flag:
                coords_list = []
                for i in range(len(ster_list)):
                    for _ in range(skip_lines_tab[i]):
                        file.readline()
                    a = file.readline()
                    a_split = a.split()
                    for j in range(ster_list[i][1]):
                        coords_list.append( eval( a_split[ (ster_list[i][2][j] - 1) ] ) )
                for i in range(n_atom_2_read):
                    at_xyz = np.array( [ coords_list[3*i], coords_list[3*i+1], coords_list[3*i+2] ], dtype=np.float64)
                    atomic_coordinates.append( at_xyz )                                       
    # use the lists generated above:           
    for i in range(n_atom_2_read):
        atom = Atom(atomic_numbers[i],atomic_coordinates[i])
        atom.set_number(at_numbers[i])
        atom_list.append(atom)            
    return atom_list


def atm_nbrs_2_seq_in_hess(two_at_tuple):
    """computes a sequential number (1-based) for xx, yx and zx hessian elements in
    a lower triangular matrix
    Input
    two_at_tuple : a tuple with two atom (1-based) numbers (int), in ascending order
    Returns
    a tuple with three int - (1-based) seqence numbers for 3x3 hessian elements begining 1st, 2nd and 3rd row
    """
    nrB = two_at_tuple[1]
    nrA = two_at_tuple[0]
    row_nr = 3 * (nrB - 1) + 1
    col_nr = 3 * (nrA - 1) + 1
    seq_xx = col_nr + int( ( (row_nr - 1) * row_nr ) / 2 )
    seq_yx = seq_xx + row_nr
    seq_zx = seq_xx + 2*row_nr + 1
    return (seq_xx, seq_yx, seq_zx)


def fchk_read_hessian(file,at_pair):
    """
    reads 3x3 partial hessian for a pair of atoms from fchk file
    Input
    file : fchk file (file object)
    at_pair : a tuple of two integers - 1-based atom numbers (in ascending order)
    Returns
    hess : 3x3 numpy array of float64 type 
    """
    seq_numbers = atm_nbrs_2_seq_in_hess(at_pair)
    line_pos_col_1 = [which_line_pos(seq_numbers[i], 5) for i in range(3)] # a list of tuples with (line_nr, position(col_nr)) in a block with 5 numbers per line
    line_pos_col_2 = [which_line_pos(seq_numbers[i]+1, 5) for i in range(3)]
    line_pos_col_3 = [which_line_pos(seq_numbers[i]+2, 5) for i in range(3)]
    line_pos_merge = [line_pos_col_1, line_pos_col_2, line_pos_col_3]
    line_pos_in_seq = []
    for i in range(3):  # order (line, pos) is sequance as these elements will be read
        for elem in line_pos_merge:
            line_pos_in_seq.append(elem[i])
    skip_lines_tab = [] # a table with number of lines to be skipped   
    for i in range(9):
        if i == 0:
            skip_lines_tab.append( line_pos_in_seq[i][0] - 1 )
        else:
            jump = ( line_pos_in_seq[i][0] - line_pos_in_seq[i-1][0] - 1 )
            if jump != -1:
                skip_lines_tab.append( jump )
    ster_list = gen_ster_list(line_pos_in_seq)      
    # w pliku file wyszukaj odp linii       
    hess_list = []   # 3x3 hessian elements in sequence (as read)     
    file.seek(0)
    flag_line = "Cartesian Force Constants"
    while True:
        try:
           a = file.readline()
        except UnicodeDecodeError:
           pass
        if not a:
            break
        elif a[0] != ' ':
            match_flag=re.search(flag_line,a)
            if match_flag:
                for i in range(len(ster_list)):
                    for _ in range(skip_lines_tab[i]):
                        file.readline()
                    a = file.readline()
                    a_split = a.split()
                    for j in range(ster_list[i][1]):
                        hess_list.append( eval( a_split[ (ster_list[i][2][j] - 1) ] ) )
            hess = np.array( [hess_list[0:3], hess_list[3:6], hess_list[6:9]] , dtype=np.float64)  
    return hess


def triple_to_2_bond_labels(triple):
    """ takes a tuple of 3 integers representing an A-B-C angle and returns
    two 2-int tuples - labels for A-B and B-C bonds"""
    a_nr = triple[0]
    b_nr = triple[1]
    c_nr = triple[2]
    ab_list = [a_nr, b_nr]
    ab_list.sort()
    ab_label = tuple( ab_list )
    bc_list = [b_nr, c_nr]
    bc_list.sort()
    bc_label = tuple( bc_list )                    
    return ab_label, bc_label           
        
        
def read_section_from_input(file, flag_line, end_line):
    """reads a section (lines) from file
    contained between line starting with flag_line
    and end_line 
    ---
    file : file object
    flag_line : string
    end_line : string
    ---
    Returns:
    section : a list with strings, each string per line
    """
    section = []
    # w pliku file wyszukaj linii zawierajacej flag_line
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            while True:
                a = file.readline()
                if not a:
                    break
                elif re.search(end_line,a):
                    break
                section.append(a)
    return section


# # FLAGS = headers that read_file is looking for in order to acquire information from given file
# FLAGS = ["Nuclear charges", "Current cartesian coordinates", "Cartesian Force Constants"]


# def read_file(file, atom_one, atom_two):
#     bool_FLAGS = [True, True, True]
#     with open(file, 'r') as file:
#         file.seek(0)
#         atom_one_charge = atom_two_charge = 0
#         atom_one_coordinates = np.empty((1, 3), dtype=np.float64)
#         atom_two_coordinates = np.empty((1, 3), dtype=np.float64)
#         maximum, minimum = max(atom_one, atom_two), min(atom_one, atom_two)  # gets a value of higher, lower atom number
#         for line in file:
#             if line[0] != ' ':
#                 if bool_FLAGS[0]:  # "Nuclear charges"
#                     words = set(line.split())  # header words
#                     flag = set(FLAGS[0].split())  # flag words
#                     if flag == flag & words:  # check if common words are the flag - Nuclear charges
#                         bool_FLAGS[0] = False  # "Nuclear charges" flag found
#                         value = 0  # value of current position- current considered atom charge
#                         '''since there are 5 values in a row indexed starting from 1, given atom number we can 
#                         determine amount of rows to skip using modulo5 and by 5 division 
#                         [ 1 2 3 4 5  ] 
#                         [ 6 7 8 9 10 ] 
#                         [ ... ] 
#                         skip1 -> minimum // 5 is an amount of rows preceding first atom charge 
#                         skip2 is much harder to determine, because there are many possible edge cases, therefore 
#                         its easier to underestimate the amount of rows skipped by subtracting 3 from most possible rows 
#                         skipped and work the way up from there reading at most 3 additional rows with 5 values 
#                         // is used instead of / since positions are int values '''
#                         skip1 = minimum // 5
#                         if minimum % 5 == 0: skip1 -= 1
#                         skip2 = (maximum - minimum) // 5 - 3
#                         if skip1 > 0:
#                             for _ in range(skip1):
#                                 next(file)  # skipping a row
#                             value += 5 * skip1  # correcting the current position value
#                         newline = file.readline()
#                         for j in newline.split():  # assessing atom_one_charge and possibly atom_two charge
#                             value += 1
#                             if value == minimum:
#                                 atom_one_charge = j
#                             elif value == maximum:
#                                 atom_two_charge = j
#                         if skip2 > 0:
#                             for _ in range(skip2):
#                                 next(file)  # skipping a row
#                             value += 5 * skip2  # correcting the current position value
#                         while atom_two_charge == 0:  # if atom_two_charge wasn't in the same row
#                             # while statement used as skip2 value is ambiguous
#                             newline = file.readline()
#                             for j in newline.split():
#                                 value += 1
#                                 if value == maximum:
#                                     atom_two_charge = j  # assessing the atom_two charge

#                 elif bool_FLAGS[1]:  # "Current cartesian coordinates"
#                     words = set(line.split())  # header words
#                     flag = set(FLAGS[1].split())  # flag words
#                     if flag == flag & words:  # check if common words are the flag - Current cartesian coordinates
#                         bool_FLAGS[1] = False  # "Nuclear charges" flag found
#                         value = 0  # value of current position- current considered atom coordinate
#                         '''
#                         since there are three coordinates for each atom and positions are indexed starting from 1,
#                         the formula 3n-2 gives a position of the x coordinate of corresponding n-th atom 
#                         skip values are ascertained similiary to "Nuclear charge" flag section, the only difference
#                         being 2 more values per atom in this section 
#                         '''
#                         atom_one_position = 3 * minimum - 2  # position of the x coordinate of the first atom
#                         atom_two_position = 3 * maximum - 2  # position of the x coordinate of the second atom
#                         skip1 = atom_one_position // 5  # lines preceding lower atom x coordinate
#                         if atom_one_position % 5 == 0: skip1 -= 1
#                         skip2 = (atom_two_position - atom_one_position) // 5 - 3  # lines between lower and higher
#                         # atom x coordinate
#                         check1 = True  # bool to make sure all three coordinates of atom one were found
#                         check2 = True  # bool to make sure all three coordinates of atom two were found
#                         if skip1 > 0:
#                             for _ in range(skip1):
#                                 next(file)  # skipping a row
#                             value += 5 * skip1  # correcting the current position value
#                         '''
#                          all three coordinates are not necessarily in the same row, one of the edge cases being:
#                         [ aa aa aa aa x1 ]
#                         [ y1 z1 x2 y2 z2 ]
#                         [ aa aa aa aa ...]
#                         its vital to consider each value for each position in not skipped rows 
#                         '''
#                         while check1:
#                             newline = file.readline()
#                             for j in newline.split():
#                                 value += 1
#                                 if value == atom_one_position:  # x1
#                                     atom_one_coordinates[0][0] = j  # inserts found coordinate to coordinates vector
#                                 elif value == atom_one_position + 1:  # y1
#                                     atom_one_coordinates[0][1] = j
#                                 elif value == atom_one_position + 2:  # z1
#                                     atom_one_coordinates[0][2] = j
#                                     check1 = False  # all coordinates of first atom found
#                                 elif value == atom_two_position:  # x2
#                                     atom_two_coordinates[0][0] = j
#                                 elif value == atom_two_position + 1:  # y2
#                                     atom_two_coordinates[0][1] = j
#                                 elif value == atom_two_position + 2:  # z2
#                                     atom_two_coordinates[0][2] = j
#                                     check2 = False  # for some cases when the numbers are one after the other, and
#                                     # coordinates of the first one are in two rows, its possible to find coordinates
#                                     # of both atoms at the same time
#                         if skip2 > 0:
#                             for _ in range(skip2):
#                                 next(file)  # skipping a row
#                             value += 5 * skip2  # correcting the current position value
#                         while check2:
#                             newline = file.readline()
#                             for j in newline.split():
#                                 value += 1
#                                 if value == atom_two_position:  # x2
#                                     atom_two_coordinates[0][0] = j
#                                 elif value == atom_two_position + 1:  # y2
#                                     atom_two_coordinates[0][1] = j
#                                 elif value == atom_two_position + 2:  # z2
#                                     atom_two_coordinates[0][2] = j
#                                     check2 = False  # all three coordinates of the second atom were found

#                 elif bool_FLAGS[2]:  # "Cartesian Force Constants
#                     words = set(line.split())  # header words
#                     flag = set(FLAGS[2].split())  # flag words
#                     if flag == flag & words:  # check if common words are the flag - Cartesian Force Constants
#                         bool_FLAGS[2] = False  # "Cartesian Force Constants" flag found
#                         value = 0  # value of current position- current considered hessian value
#                         '''
#                              x1  y1  z1  x2  y2  z2  x3 y3
#                              _   _   _   _   _   _   _  _ 
#                         x1 [ 1 
#                         y1 [ 2   3 
#                         z1 [ 4   5   6 
#                         x2 [ 7   8   9   10
#                         y2 [ 11  12  13  14  15
#                         z2 [ 16  17  18  19  20  21
#                         x3 [ 22  23  24  25  26  27  28
#                         y3 [ ...
#                         The key to get right values in hessian is establishing the position of a first value in said
#                         hessian. Dividing 3 from the bigger atom number multiplied by 3 provides a shift value - 
#                         number of rows preceding the row with the first hessian value. Then, 3n - 2 formula used
#                         smaller atom number gives us the right position within the row. To jump from the first position
#                         to fourth is a matter of adding a shift value plus one- since shift is a number of elements in
#                         last skipped row. Similarly jump from fourth to seventh position can be made.  
#                         '''
#                         shift = 3 * maximum - 3  # number of rows skipped
#                         # positions of atoms in file corresponding to hessian matrix:
#                         first_position = (3 * minimum - 2) + sum(k for k in range(1, shift + 1))
#                         fourth_position = first_position + shift + 1
#                         seventh_position = fourth_position + shift + 2
#                         hessian = [first_position, first_position + 1, first_position + 2,
#                                    fourth_position, fourth_position + 1, fourth_position + 2,
#                                    seventh_position, seventh_position + 1, seventh_position + 2]
#                         hessian_values = np.empty((3, 3),
#                                                   dtype=np.float64)  # empty float 3x3 matrix to collect actual hessian
#                         position = 0  # itterates through positions in matrix
#                         hessian_value = first_position  # first position that will be looked for
#                         skip1 = first_position // 5  # amount of rows before first position
#                         if first_position % 5 == 0: skip1 -= 1
#                         if skip1 > 0:
#                             for _ in range(skip1):
#                                 next(file)  # skipping a row
#                             value += 5 * skip1  # correcting the current position value
#                         while position < 9:  # as long as all 9 values are not found
#                             newline = file.readline()
#                             for j in newline.split():
#                                 value += 1
#                                 if value == hessian_value:
#                                     insert_hessian(position, j, hessian_values)  # insert value to hessian_value
#                                     position += 1  # gets new position
#                                     if position < 9: hessian_value = hessian[position]  # gets new position

#                         a1 = make_atom(atom_one_charge, atom_one_coordinates)  # create atom one
#                         a2 = make_atom(atom_two_charge, atom_two_coordinates)  # create atom two
#                         if atom_one > atom_two:  # if atoms in a given file are in an order: higher value, lower value-
#                             # its necessary to switch found values afterwards
#                             temp = a1
#                             a1 = a2
#                             a2 = temp
#                         bond = make_bond(a1, a2, hessian_values)  # create bond
#                         file.close()
#                         return a1, a2, bond  # return [object(a1), object(a2), object(bond)]
#     print("file error")  # the function executed on a well-structured file should never reach this point
#     file.close()


# # read_file collects information about charges, cartesian coordinates and hessian between two given atoms

# '''
# The norm of a vector is zero if and only if the vector is a zero vector. All possible atom positions require the atoms
# to be in different positions so vector between them will never be zero- there is no need for checking whether the norm
# differs from 0 when dividing.  
# '''

# # bond_info calls read_file and determines bond force constant among other figures
# def bond_info(file, atom_one, atom_two):
#     # result  =  [ atom one, atom two,   bond one-two   ]
#     result = read_file(file, atom_one, atom_two)

#     w, v = np.linalg.eig(-1. * result[2].kcm_hessian)  # eigenvalues and eigenvectors
#     diff_u_ab = result[1].angs_coordinates - result[0].angs_coordinates  # difference in B, A coordinates
#     u_ab = diff_u_ab / np.linalg.norm(diff_u_ab)  # normalized vector
#     k = sum(w[i] * np.abs(np.dot(u_ab, v[:, i])) for i in range(3))  # k bond force constant for first hessian

#     w2, v2 = np.linalg.eig(-1. * result[2].kcm_hessian2)  # second eigenvalues and eigenvectors
#     k2 = sum(w2[i] * np.abs(np.dot(u_ab, v2[:, i])) for i in range(3))  # k bond force constant for second hessian
#     '''
#     print(w[0])
#     print(np.abs(np.dot(u_ab,v[:,0])))
#     print(w[0] * np.abs(np.dot(u_ab,v[:,0])))
#     print(w[1])
#     print(np.abs(np.dot(u_ab,v[:,1])))
#     print(w[0] * np.abs(np.dot(u_ab,v[:,1])))
#     print(w[2])
#     print(np.abs(np.dot(u_ab,v[:,2])))
#     print(w[2] * np.abs(np.dot(u_ab,v[:,2])))
#     '''

#     K = (k + k2) / 2  # mean k force constant for two calculated values
#     distance = np.linalg.norm(diff_u_ab)  # distance AB
#     return [result[0],result[1],distance,K]
#     #return [atom A, atom B, distance AB, k_force_constant]

# # angle_info calls read_file for atoms (1,2), (2,3) and determines bond angle force constant among other figures
# def angle_info(file, atom_one, atom_two, atom_three):
#     # result  =  [ atom one, atom two,   bond one-two   ]
#     # result2 =  [ atom two, atom three, bond two-three ]
#     result = read_file(file, atom_one, atom_two)
#     result2 = read_file(file, atom_two, atom_three)

#     # eigenvalues and eigenvectors for A-B, B-C hessians
#     w, v = np.linalg.eig(-1. * result[2].kcm_hessian)
#     w2, v2 = np.linalg.eig(-1. * result2[2].kcm_hessian)

#     # eigenvalues and eigenvectors for transposed A-B, B-C hessians
#     w3, v3 = np.linalg.eig(-1. * result[2].kcm_hessian2)
#     w4, v4 = np.linalg.eig(-1. * result2[2].kcm_hessian2)


#     diff_u_ab = result[1].angs_coordinates - result[0].angs_coordinates  # difference in coordinates B - A
#     u_ab = diff_u_ab / np.linalg.norm(diff_u_ab)

#     #diff_u_ba = result[0].angs_coordinates - result[1].angs_coordinates  # A - B
#     #u_ba = diff_u_ba / np.linalg.norm(diff_u_ba)


#     diff_u_cb = result2[0].angs_coordinates - result2[1].angs_coordinates  # difference in coordinates B - C
#     u_cb = diff_u_cb / np.linalg.norm(diff_u_cb)

#     #diff_u_bc = result2[1].angs_coordinates - result2[0].angs_coordinates  # C - B
#     #u_bc = diff_u_bc / np.linalg.norm(diff_u_bc)


#     cross = np.cross(u_cb, u_ab)  # cross product of u_cb and u_ab vectors
#     cross_norm = cross / np.linalg.norm(cross)  # normalised u_n
#     u_pa = np.cross(cross_norm, u_ab)  # u_pa = u_n x u_ab
#     u_pc = np.cross(u_cb, cross_norm)  # u_pc = u_cb x u_n
#     R_ab_sq = np.linalg.norm(diff_u_ab) ** 2  # length of bond ab squared
#     R_cb_sq = np.linalg.norm(diff_u_cb) ** 2  # length of bond cb squared

#     # one over k formula
#     # one_over_kt =\
#     # 1 / (2 * (R_ab_sq * sum(w[i] *  np.abs(np.dot(u_pa, v[:, i])) for i in range(3)))) +\
#     # 1 / (2 * (R_ab_sq * sum(w3[i] * np.abs(np.dot(u_pa, v3[:, i])) for i in range(3)))) +\
#     # 1 / (2 * (R_cb_sq * sum(w2[i] * np.abs(np.dot(u_pc, v2[:, i])) for i in range(3)))) +\
#     # 1 / (2 * (R_cb_sq * sum(w4[i] * np.abs(np.dot(u_pc, v4[:, i])) for i in range(3))))
    
#     one_over_kt =\
#     ( 2 / ( (R_ab_sq * sum(w[i] *  np.abs(np.dot(u_pa, v[:, i])) for i in range(3))) +\
#     ( R_ab_sq * sum(w3[i] * np.abs(np.dot(u_pa, v3[:, i])) for i in range(3))) ) ) +\
#     ( 2 / ( (R_cb_sq * sum(w2[i] * np.abs(np.dot(u_pc, v2[:, i])) for i in range(3))) +\
#     ( R_cb_sq * sum(w4[i] * np.abs(np.dot(u_pc, v4[:, i])) for i in range(3))) ) )
#     k_theta = 1 / one_over_kt  # angle force constant

#     # angle ABC in degrees
#     angle = np.degrees(np.arccos(u_ab[0][0] * u_cb[0][0] + u_ab[0][1] * u_cb[0][1] + u_ab[0][2] * u_cb[0][2]))
#     '''
#     one_over_kt = 1 / (R_ab_sq * sum(
#     (w[i] / 2) * np.abs(np.dot(u_pa, v[:, i])) + (w3[i] / 2) * np.abs(np.dot(u_pa, v3[:, i])) for i in range(3))) +\
#                 1 / (R_cb_sq * sum(
#     (w2[i] / 2) * np.abs(np.dot(u_pc, v2[:, i])) + (w4[i] / 2) * np.abs(np.dot(u_pc, v4[:, i])) for i in range(3)))
#     R_ab_sq * sum((w[i]/2) * np.abs(np.dot(u_pa, v[:, i])) + (w3[i]/2) * np.abs(np.dot(u_pa, v3[:, i])))
#     R_cb_sq * sum((w2[i]/2) * np.abs(np.dot(u_pa, v2[:, i])) + (w4[i]/2) * np.abs(np.dot(u_pa, v4[:, i])))
#     '''

#     return (result[0],result[1],result2[1],angle,k_theta)
#     #return [atom1, atom2, atom3, angle_in_degrees, angle_force_constant]

#     # driver code
# with open(file_name, 'r') as input_file:
#     output_file = open(file_name2, "w+")
#     commands = input_file.readline()
#     for c in range(int(commands)):  # for each command in input file
#         command = input_file.readline().split()  # ascertain between bond/angle
#         if command[1] == "bond":  # bond_info (file_name, atom_one, atom_two)
#             bond = bond_info(command[0], int(command[2]), int(command[3]))
#             # write into an output: no. file_name bond atom1.symbol-atom2.symbol distance12 bond_force_constant
#             output_file.write(str(c+1) + " " + command[0] + " " + command[1] + " " + str(bond[0].symbol) +
#                               "-" + str(bond[1].symbol) + " " + str(bond[2]) + " " + str(bond[3][0]) + "\n")
#         elif command[1] == "angle":  # angle_info (file_name, atom_one, atom_two (the atom in the middle), atom_three)
#             angle = angle_info(command[0], int(command[2]), int(command[3]), int(command[4]))
#             # write into an output: no. file_name angle atom1.symbol-atom2.symbol-atom3symbol
#             # angle123_in_degrees angle_force_co nstant
#             output_file.write(str(c+1) + " " + command[0] + " " + command[1] + " " + str(angle[0].symbol) +
#                               "-" + str(angle[1].symbol) + "-" + str(angle[2].symbol) + " "
#                               + str(angle[3]) + " " + str(angle[4][0]) + "\n")
#         else:
#             print("line ", c, ": incorrect command name")
#     input_file.close()
#     output_file.close()

