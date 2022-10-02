"""
Tools for building side chains with templates and given chi angles
Some of the functions borrowed from IDP Conformer Generator package (https://github.com/julie-forman-kay-lab/IDPConformerGenerator) developed by Joao M. C. Teixeira

Coded by Jie Li
Jul 27, 2021
"""

import numpy as np
from mcsce.libs.libcalc import calc_angle, calc_torsion_angles, rotate_coordinates_Q
from mcsce.core.build_definitions import sidechain_templates





def rotate_tor(original_tor, tor, unit_vector, coords, offset):
    """
    Rotate a torsion angle from the original value to the target value along a specific axis 
    (defined by unit_vector) for all atoms in the coords, and apply the offset

    Parameters
    ----------
    original_tor : np.float
        original torsion angle in template.
    tor : np.float
        target torsion angle.
    unit_vector : numpy nd.array, shape (3, ), dtype=float64
        Unitary vector of the axis to rotate around.
    coords : numpy nd.array, shape (N, 3), dtype=float64
        Sidechain coordinates to rotate.
    offset : numpy nd.array, shape (3, ), dtype=float64
        Coordinate of the atom at rotation axis.

    Returns
    -------
    Rotated coordinates.

    Credit
    -------
    Written by Oufan Zhang

    """   
    displaced = coords - offset

    # rotate
    rot_angle = original_tor - tor
    if rot_angle > np.pi:
        rot_angle -= 2*np.pi
    elif rot_angle < -np.pi:
        rot_angle += 2*np.pi
    rotated = rotate_coordinates_Q(displaced, -unit_vector, rot_angle)
    
    return rotated + offset

def rotate_sidechain(res_type, tors):
    """
    Function for rotating side chain torsion angles according to privided tors

    Parameters
    ----------
    res_type : str 
        three letter coding of amino acid
    tors : numpy nd.array, shape (M, ), dtype=float64
        M depends on res_type, each angle [-180, 180]
        The chi torsion angles in degrees.

    Returns
    -------
    Rotated sidechain coordinates

    Credit
    -------
    Original version written by Oufan Zhang, modified by Jie Li

    """
    assert not res_type in ['ALA', 'GLY', 'PCA']

    # convert degrees to radians
    tors = tors * np.pi / 180

    # fetch side chain structures and atom labels
    template_structure, sidechain_idx = sidechain_templates[res_type]
    sidechain_label = template_structure.data_array[:, 2] # atom name column
    template = template_structure.coords
        
    # common atom template index
    N_idx = 0
    CA_idx = 1
    CBs_idx = 4 #CB, CB1. CB2
    CGs_idx = 5 #OG, CG, CG1, OG1, SG
    CDs_idx = 6 #SD, CD, CD1, ND1, OD1, CS, P 
    CEs_idx = 7 #CE, NE, CE1, CE2, OE11, OE1, NE1
    CZs_idx = 8 #CZ, NZ
    NH_idx = 9  #NH1, CH
    N_CA_CB_CG = [N_idx, CA_idx, CBs_idx, CGs_idx]
    CA_CB_CG_CD = [CA_idx, CBs_idx, CGs_idx, CDs_idx]
    CB_CG_CD_CE = [CBs_idx, CGs_idx, CDs_idx, CEs_idx] 
    CG_CD_CE_CZ = [CGs_idx, CDs_idx, CEs_idx, CZs_idx]
    CD_CE_CZ_NH = [CDs_idx, CEs_idx, CZs_idx, NH_idx]

    # place CB at origin for rotation, exclude bb atoms and HA
    ori_chi1 = calc_torsion_angles(template[N_CA_CB_CG, :])[0]  # The original chi1 torsion angle value
    CB = template[4]
    unit_vector = CB/np.linalg.norm(CB)
    HA = np.where(np.isin(sidechain_label, ['HA','CB2','HB21','HB22','HB23']))[0] #MGN 
    chi1_idx = np.delete(sidechain_idx, np.where(np.isin(sidechain_idx, HA))[0])
    template[chi1_idx, :] = rotate_tor(ori_chi1, tors[0], unit_vector, 
                                             template[chi1_idx, :], CB)
    if len(tors) == 1:
        return template, sidechain_idx
    
    ori_chi2 = calc_torsion_angles(template[CA_CB_CG_CD, :])[0]
    CGs = template[5]
    unit_vector = (CGs - CB)/np.linalg.norm(CGs - CB)
    HBs = np.where(np.isin(sidechain_label, ['HB','HB2','HB3','CB','CG2','HG21','HG22','HG23','CB1','HB12','HB13','OB']))[0] #MGN BHD
    chi2_idx = np.delete(chi1_idx, np.where(np.isin(chi1_idx, HBs))[0])
    template[chi2_idx, :] = rotate_tor(ori_chi2, tors[1], unit_vector, 
                                             template[chi2_idx, :], CGs)
    #print('rotated chi2:', np.degrees(calc_torsion_angles(template[CA_CB_CG_CD, :])[0]))
    if len(tors) == 2:
        return template, sidechain_idx
    
    # special ptm case 2+2
    if res_type in ['Y1P', 'PTR']:
        # chi3
        CE_CZ_OH_P = [9, 10, 11, 12]
        ori_chi3 = calc_torsion_angles(template[CE_CZ_OH_P, :])[0] 
        unit_vector = (template[11] - template[10])/np.linalg.norm(template[11] - template[10])
        chi3_idx = [12, 13, 14, 15]
        if res_type == 'Y1P': chi3_idx += [-1]
        template[chi3_idx, :] = rotate_tor(ori_chi3, tors[2], unit_vector,
                                             template[chi3_idx, :], template[11])
        if len(tors) == 3: return template, sidechain_idx
        # chi4
        CZ_OH_P_OP = [10, 11, 12, 13]
        ori_chi4 = calc_torsion_angles(template[CZ_OH_P_OP, :])[0]  
        unit_vector = (template[12] - template[11])/np.linalg.norm(template[12] - template[11])
        chi4_idx = [13, 14, 15]
        if res_type == 'Y1P': chi4_idx += [-1]
        template[chi4_idx, :] = rotate_tor(ori_chi4, tors[3], unit_vector,
                                             template[chi4_idx, :], template[12])
        return template, sidechain_idx

    if res_type in ['H1D', 'H2D']:
        # chi3
        CG_ND_P_OP = [5, 7, 10, 11]
        ori_chi3 = calc_torsion_angles(template[CG_ND_P_OP, :])[0]  
        unit_vector = (template[10] - template[7])/np.linalg.norm(template[10] - template[7])
        chi3_idx = [11, 12, 13]
        if res_type == 'H1D': chi3_idx += [-1]
        template[chi3_idx, :] = rotate_tor(ori_chi3, tors[2], unit_vector,
                                             template[chi3_idx, :], template[10])
        return template, sidechain_idx

    if res_type in ['H1E', 'H2E']:
        # chi3
        CD_NE_P_OP = [6, 9, 10, 11]
        ori_chi3 = calc_torsion_angles(template[CD_NE_P_OP, :])[0]  
        unit_vector = (template[10] - template[9])/np.linalg.norm(template[10] - template[9])
        chi3_idx = [11, 12, 13]
        if res_type == 'H1D': chi3_idx += [-1]
        template[chi3_idx, :] = rotate_tor(ori_chi3, tors[2], unit_vector,
                                             template[chi3_idx, :], template[10])
        return template, sidechain_idx

    ori_chi3 = calc_torsion_angles(template[CB_CG_CD_CE, :])[0]
    CDs = template[6]
    unit_vector = (CDs - CGs)/np.linalg.norm(CDs - CGs)
    HGs = ['HG','HG2','HG3','CG','CG1']
    if res_type == 'MEN': HGs += ['OD1']
    elif res_type == 'CGU': HGs += ['OE21','OE22','CD2']
    elif res_type in ['SEP', 'S1P']: HGs += ['OG']
    elif res_type in ['TPO', 'T1P']: HGs += ['OG1']
    HGs_idx = np.where(np.isin(sidechain_label, HGs))[0]
    chi3_idx = np.delete(chi2_idx, np.where(np.isin(chi2_idx, HGs_idx))[0])
    template[chi3_idx, :] = rotate_tor(ori_chi3, tors[2], unit_vector, 
                                             template[chi3_idx, :], CDs)
    #print('rotated chi3:', np.degrees(calc_torsion_angles(template[CB_CG_CD_CE, :])[0]))
    if len(tors) == 3:
        return template, sidechain_idx

    ori_chi4 = calc_torsion_angles(template[CG_CD_CE_CZ, :])[0]
    CEs = template[7]
    unit_vector = (CEs - CDs)/np.linalg.norm(CEs - CDs)
    HDs = ['HD','HD2','HD3','CD','SD']
    if res_type == 'LYZ': HDs += ['OH','HH']
    elif res_type == 'AGM': HDs += ['CE2','HE21','HE22','HE23']
    HDs_idx = np.where(np.isin(sidechain_label, HDs))[0]
    chi4_idx = np.delete(chi3_idx, np.where(np.isin(chi3_idx, HDs_idx))[0])
    template[chi4_idx, :] = rotate_tor(ori_chi4, tors[3], unit_vector, 
                                             template[chi4_idx, :], CEs)
    #print('rotated chi4:', np.degrees(calc_torsion_angles(template[CG_CD_CE_CZ, :])[0]))
    if len(tors) == 4:
        return template, sidechain_idx
    
    ori_chi5 = calc_torsion_angles(template[CD_CE_CZ_NH, :])[0]
    CZ = template[8]
    unit_vector = (CZ - CEs)/np.linalg.norm(CZ - CEs)
    HEs = np.where(np.isin(sidechain_label, ['HE','HE1','HE2','HE3','CD']))[0] #AGM ALY 
    chi5_idx = np.delete(chi4_idx, np.where(np.isin(chi4_idx, HEs))[0])
    template[chi5_idx, :] = rotate_tor(ori_chi5, tors[4], unit_vector, 
                                             template[chi5_idx, :], CZ)
    #print('rotated chi5:', np.degrees(calc_torsion_angles(template[CD_CE_CZ_NH, :])[0]))
 
    return template, sidechain_idx


if __name__ == "__main__":
    #ARG
    template_coord, _ = rotate_sidechain("H1D", np.array([-123, 173, 85]))
    arg_temp = sidechain_templates["H1D"][0]
    arg_temp.coords = template_coord
    arg_temp.write_PDB("rotated_H1D.pdb")
