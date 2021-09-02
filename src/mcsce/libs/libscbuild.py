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
    assert not res_type in ['ALA', 'GLY']

    # convert degrees to radians
    tors = tors * np.pi / 180

    # fetch side chain structures and atom labels
    template_structure, sidechain_idx = sidechain_templates[res_type]
    sidechain_label = template_structure.data_array[:, 2] # atom name column
    template = template_structure.coords
    
    # chi1 angles in template
    if res_type == 'SER':
        N_CA_CB_CG = [np.where(sidechain_label=='N')[0][0], 
                               np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='OG')[0][0]]
    elif res_type == 'THR':
        N_CA_CB_CG = [np.where(sidechain_label=='N')[0][0], 
                               np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='OG1')[0][0]]
    elif res_type in ['ILE', 'VAL']:
        N_CA_CB_CG = [np.where(sidechain_label=='N')[0][0], 
                               np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG1')[0][0]]
    elif res_type == "CYS":
        N_CA_CB_CG = [np.where(sidechain_label=='N')[0][0], 
                               np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='SG')[0][0]]
    else:
        N_CA_CB_CG = [np.where(sidechain_label=='N')[0][0], 
                               np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0]]
    ori_chi1 = calc_torsion_angles(template[N_CA_CB_CG, :])[0]  # The original chi1 torsion angle value
    
    # chi2 angles in template
    if res_type == 'MET':       
        CA_CB_CG_CD = [np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='SD')[0][0]]
        CB_CG_CD_CE = [np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='SD')[0][0],
                               np.where(sidechain_label=='CE')[0][0]]
    elif res_type == 'ILE':
        CA_CB_CG_CD = [np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG1')[0][0],
                               np.where(sidechain_label=='CD1')[0][0]] 
    elif res_type in ['HIS', 'HIP', 'HID', 'HIE']:
        CA_CB_CG_CD = [np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='ND1')[0][0]]
    elif res_type in [ 'ASN', 'ASP']:
        CA_CB_CG_CD = [np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='OD1')[0][0]]
    elif res_type in ['TRP', 'PHE', 'LEU', 'TYR']:
        CA_CB_CG_CD = [np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='CD1')[0][0]]
    elif res_type not in ['SER', 'THR', 'VAL', 'CYS']:
        CA_CB_CG_CD = [np.where(sidechain_label=='CA')[0][0],
                               np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='CD')[0][0]]
    
    # rest of residues
    if res_type in ['GLN','GLU']:
        CB_CG_CD_CE = [np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='CD')[0][0],
                               np.where(sidechain_label=='OE1')[0][0]]
    elif res_type == 'LYS':
        CB_CG_CD_CE = [np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='CD')[0][0],
                               np.where(sidechain_label=='CE')[0][0]]
        CG_CD_CE_CZ = [np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='CD')[0][0],
                               np.where(sidechain_label=='CE')[0][0],
                               np.where(sidechain_label=='NZ')[0][0]]
    elif res_type == 'ARG':
        CB_CG_CD_CE = [np.where(sidechain_label=='CB')[0][0],
                               np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='CD')[0][0],
                               np.where(sidechain_label=='NE')[0][0]]
        CG_CD_CE_CZ = [np.where(sidechain_label=='CG')[0][0],
                               np.where(sidechain_label=='CD')[0][0],
                               np.where(sidechain_label=='NE')[0][0],
                               np.where(sidechain_label=='CZ')[0][0]]
        CD_NE_CZ_NH1 = [np.where(sidechain_label=='CD')[0][0],
                               np.where(sidechain_label=='NE')[0][0],
                               np.where(sidechain_label=='CZ')[0][0],
                               np.where(sidechain_label=='NH1')[0][0]]
        ori_chi5 = calc_torsion_angles(template[CD_NE_CZ_NH1, :])[0]
        
    
    # place CB at origin for rotation, exclude bb atoms and HA
    CB = template[np.where(sidechain_label=='CB')[0][0]]
    unit_vector = CB/np.linalg.norm(CB)
    # HA = np.where(sidechain_label=='HA')[0][0]
    # chi1_idx = np.delete(sidechain_idx, np.where(sidechain_idx==HA)[0][0])
    chi1_idx = sidechain_idx
    template[chi1_idx, :] = rotate_tor(ori_chi1, tors[0], unit_vector, 
                                             template[chi1_idx, :], CB)
    if len(tors) == 1:
        return template, sidechain_idx
    
    ori_chi2 = calc_torsion_angles(template[CA_CB_CG_CD, :])[0]
    CGs = template[5]
    assert sidechain_label[5] in ['CG','CG1']
    unit_vector = (CGs - CB)/np.linalg.norm(CGs - CB)
    HBs = np.where(np.isin(sidechain_label, ['HB','HB2','HB3','CB','CG2','HG21','HG22','HG23']))[0]
    chi2_idx = np.delete(chi1_idx, np.where(np.isin(chi1_idx, HBs))[0])
    template[chi2_idx, :] = rotate_tor(ori_chi2, tors[1], unit_vector, 
                                             template[chi2_idx, :], CGs)
    if len(tors) == 2:
        return template, sidechain_idx
    
    ori_chi3 = calc_torsion_angles(template[CB_CG_CD_CE, :])[0]
    CDs = template[6]
    assert sidechain_label[6] in ['CD','SD']
    unit_vector = (CDs - CGs)/np.linalg.norm(CDs - CGs)
    HGs = np.where(np.isin(sidechain_label, ['HG','HG2','HG3','CG','CG1']))[0]
    chi3_idx = np.delete(chi2_idx, np.where(np.isin(chi2_idx, HGs))[0])
    template[chi3_idx, :] = rotate_tor(ori_chi3, tors[2], unit_vector, 
                                             template[chi3_idx, :], CDs)
    if len(tors) == 3:
        #print('rotated chi3:', np.degrees(calc_torsion_angles(template[CB_CG_CD_CE, :])[0]))
        return template, sidechain_idx
    
    ori_chi4 = calc_torsion_angles(template[CG_CD_CE_CZ, :])[0]
    CEs = template[7]
    assert sidechain_label[7] in ['NE','CE']
    unit_vector = (CEs - CDs)/np.linalg.norm(CEs - CDs)
    HDs = np.where(np.isin(sidechain_label, ['HD','HD2','HD3','CD','SD']))[0]
    chi4_idx = np.delete(chi3_idx, np.where(np.isin(chi3_idx, HDs))[0])
    template[chi4_idx, :] = rotate_tor(ori_chi4, tors[3], unit_vector, 
                                             template[chi4_idx, :], CEs)
    if len(tors) == 4:
        return template, sidechain_idx
    
    CZ = template[8]
    unit_vector = (CZ - CEs)/np.linalg.norm(CZ - CEs)
    assert sidechain_label[19] == 'HE'
    chi5_idx = np.delete(chi4_idx, np.where(np.isin(chi4_idx, [19, 7]))[0])
    template[chi5_idx, :] = rotate_tor(ori_chi5, tors[4], unit_vector, 
                                             template[chi5_idx, :], CZ)
    
    return template, sidechain_idx


if __name__ == "__main__":
    #ARG
    template_coord, _ = rotate_sidechain("GLN", np.array([123, 35, -162]))
    arg_temp = sidechain_templates["GLN"][0]
    arg_temp.coords = template_coord
    arg_temp.write_PDB("rotated_GLN.pdb")
