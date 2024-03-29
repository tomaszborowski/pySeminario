#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a collection of classes and functions for pyseminario

Authors: Szymon Szrajer, Zuzanna Wojdyla, Tomasz Borowski

Last update: 7.09.2022
Last update: 18.05.2023
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
                           if j+1 == which_position[atom_counter] and line_counter == which_line[atom_counter]:
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

def print_help():  
    help_text = """
A python3 script computing bond and valence angle force constants using the Seminario (projected Hessian) method.

Usage example: python3 pyseminario.py pysem.inp > pysem.out

Content of the example input file (pysem.inp):

%FILES h2o_fq_nosym.fchk %END

%BONDS 3 2 %END

%ANGLES 3 1 2 %END

Comments on the format of the input file:

In the section %FILES <-> %END provide filenames (one per line; optionaly with path if not in the same directory) of fchk files (Gaussian) containing a Hessian

In the section %BONDS <-> %END provide pairs (one per line) of atom numbers for bonds for which force constant should be computed

In the section %ANGLES <-> %END provide triples (one per line) of atom numbers for angles for which force constant should be computed
    """
    
    print(help_text)
