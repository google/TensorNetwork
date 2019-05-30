# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ncon-versions of binary mera functions 
"""
import tensorflow as tf
import tensornetwork as tn



@tf.contrib.eager.defun
def left_ascending_super_operator(ham, isometry, unitary):
    inds_right_ul = [1, 2, 13, 5]
    inds_right_ur = [6, 9, 7, 16]
    inds_right_ul_c = [3, 4, 14, 10]
    inds_right_ur_c = [8, 9, 11, 17]

    inds_right_iso_l = [12, 13, -4]
    inds_right_iso_c = [5, 7, -5]
    inds_right_iso_r = [16, 15, -6]
    inds_right_iso_l_c = [12, 14, -1]
    inds_right_iso_c_c = [10, 11, -2]
    inds_right_iso_r_c = [17, 15, -3]

    inds_right_ham = [3, 4, 8, 1, 2, 6]

    hright = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_right_iso_l, inds_right_iso_c, inds_right_iso_r,
        inds_right_iso_l_c, inds_right_iso_c_c, inds_right_iso_r_c,
        inds_right_ul, inds_right_ur, inds_right_ul_c, inds_right_ur_c,
        inds_right_ham
    ])
    return hright


@tf.contrib.eager.defun
def right_ascending_super_operator(ham, isometry, unitary):
    inds_left_ul = [8, 6, 13, 7]
    inds_left_ur = [1, 2, 5, 16]
    inds_left_ul_c = [8, 9, 14, 10]
    inds_left_ur_c = [3, 4, 11, 17]
    inds_left_iso_l = [12, 13, -4]
    inds_left_iso_c = [7, 5, -5]
    inds_left_iso_r = [16, 15, -6]
    inds_left_iso_l_c = [12, 14, -1]
    inds_left_iso_c_c = [10, 11, -2]
    inds_left_iso_r_c = [17, 15, -3]
    inds_left_ham = [9, 3, 4, 6, 1, 2]

    hleft = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_left_iso_l, inds_left_iso_c, inds_left_iso_r, inds_left_iso_l_c,
        inds_left_iso_c_c, inds_left_iso_r_c, inds_left_ul, inds_left_ur,
        inds_left_ul_c, inds_left_ur_c, inds_left_ham
    ])
    return hleft


@tf.contrib.eager.defun
def get_env_disentangler_1(ham, rho, isometry, unitary):
    inds_1_rho = [17, 14, 11, 19, 12, 13]
    inds_1_wl = [16, -3, 17]
    inds_1_wc = [-4, 15, 14]
    inds_1_wr = [9, 8, 11]
    inds_1_wl_c = [16, 18, 19]
    inds_1_wc_c = [3, 5, 12]
    inds_1_wr_c = [10, 8, 13]
    inds_1_ur = [6, 7, 15, 9]
    inds_1_ul_c = [1, 2, 18, 3]
    inds_1_ur_c = [4, 7, 5, 10]
    inds_1_ham = [1, 2, 4, -1, -2, 6]

    env_1 = tn.ncon([
        rho, isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_1_rho, inds_1_wl, inds_1_wc, inds_1_wr, inds_1_wl_c, inds_1_wc_c,
        inds_1_wr_c, inds_1_ur, inds_1_ul_c, inds_1_ur_c, inds_1_ham
    ])
    return env_1

@tf.contrib.eager.defun    
def get_env_disentangler_2(ham, rho, isometry, unitary):
    inds_2_rho = [17, 14, 11, 18, 13, 12]
    inds_2_wl = [16, -3, 17]
    inds_2_wc = [-4, 15, 14]
    inds_2_wr = [6, 5, 11]
    inds_2_wl_c = [16, 19, 18]
    inds_2_wc_c = [10, 8, 13]
    inds_2_wr_c = [7, 5, 12]

    inds_2_ur = [1, 2, 15, 6]
    inds_2_ul_c = [-1, 9, 19, 10]
    inds_2_ur_c = [3, 4, 8, 7]
    inds_2_ham = [9, 3, 4, -2, 1, 2]

    env_2 = tn.ncon([
        rho, isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_2_rho, inds_2_wl, inds_2_wc, inds_2_wr, inds_2_wl_c, inds_2_wc_c,
        inds_2_wr_c, inds_2_ur, inds_2_ul_c, inds_2_ur_c, inds_2_ham
    ])
    return env_2

@tf.contrib.eager.defun    
def get_env_disentangler_3(ham, rho, isometry, unitary):

    inds_3_rho = [11, 18, 17, 12, 13, 14]
    inds_3_wl = [5, 6, 11]
    inds_3_wc = [19, -3, 18]
    inds_3_wr = [-4, 16, 17]
    inds_3_wl_c = [5, 7, 12]
    inds_3_wc_c = [8, 10, 13]
    inds_3_wr_c = [15, 16, 14]
    inds_3_ul = [1, 2, 6, 19]
    inds_3_ul_c = [3, 4, 7, 8]
    inds_3_ur_c = [9, -2, 10, 15]
    inds_3_ham = [3, 4, 9, 1, 2, -1]

    env_3 = tn.ncon([
        rho, isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_3_rho, inds_3_wl, inds_3_wc, inds_3_wr, inds_3_wl_c, inds_3_wc_c,
        inds_3_wr_c, inds_3_ul, inds_3_ul_c, inds_3_ur_c, inds_3_ham
    ])
    return env_3

@tf.contrib.eager.defun    
def get_env_disentangler_4(ham, rho, isometry, unitary):

    inds_4_rho = [11, 14, 17, 12, 13, 18]
    inds_4_wl = [8, 9, 11]
    inds_4_wc = [15, -3, 14]
    inds_4_wr = [-4, 16, 17]
    inds_4_wl_c = [8, 10, 12]
    inds_4_wc_c = [5, 3, 13]
    inds_4_wr_c = [19, 16, 18]
    inds_4_ul = [6, 7, 9, 15]
    inds_4_ul_c = [6, 4, 10, 5]
    inds_4_ur_c = [1, 2, 3, 19]
    inds_4_ham = [4, 1, 2, 7, -1, -2]

    env_4 = tn.ncon([
        rho, isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_4_rho, inds_4_wl, inds_4_wc, inds_4_wr, inds_4_wl_c, inds_4_wc_c,
        inds_4_wr_c, inds_4_ul, inds_4_ul_c, inds_4_ur_c, inds_4_ham
    ])
    return env_4



@tf.contrib.eager.defun
def get_env_isometry_1(ham, rho, isometry, unitary):
    inds_1_wc = [5, 6, 15]
    inds_1_wr = [13, 12, 16]

    inds_1_wl_c = [-1, 20, 19]
    inds_1_wc_c = [10, 11, 17]
    inds_1_wr_c = [14, 12, 18]

    inds_1_ul = [1, 2, -2, 5]
    inds_1_ur = [7, 9, 6, 13]

    inds_1_ul_c = [3, 4, 20, 10]
    inds_1_ur_c = [8, 9, 11, 14]

    inds_1_ham = [3, 4, 8, 1, 2, 7]
    inds_1_rho = [-3, 15, 16, 19, 17, 18]

    env_1 = tn.ncon([
        isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham, rho
    ], [
        inds_1_wc, inds_1_wr, inds_1_wl_c, inds_1_wc_c, inds_1_wr_c, inds_1_ul,
        inds_1_ur, inds_1_ul_c, inds_1_ur_c, inds_1_ham, inds_1_rho
    ])
    return env_1

@tf.contrib.eager.defun
def get_env_isometry_2(ham, rho, isometry, unitary):

    inds_2_wc = [14, 8, 9]
    inds_2_wr = [6, 5, 10]

    inds_2_wl_c = [-1, 19, 20]
    inds_2_wc_c = [18, 12, 13]
    inds_2_wr_c = [7, 5, 11]

    inds_2_ul = [16, 15, -2, 14]
    inds_2_ur = [1, 2, 8, 6]

    inds_2_ul_c = [16, 17, 19, 18]
    inds_2_ur_c = [3, 4, 12, 7]

    inds_2_ham = [17, 3, 4, 15, 1, 2]
    inds_2_rho = [-3, 9, 10, 20, 13, 11]

    env_2 = tn.ncon([
        isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham, rho
    ], [
        inds_2_wc, inds_2_wr, inds_2_wl_c, inds_2_wc_c, inds_2_wr_c, inds_2_ul,
        inds_2_ur, inds_2_ul_c, inds_2_ur_c, inds_2_ham, inds_2_rho
    ])
    return env_2

@tf.contrib.eager.defun
def get_env_isometry_3(ham, rho, isometry, unitary):
    """
    FIXME: this has a D^10 contraction in it! 
    """
    inds_3_wl = [5, 6, 9]
    inds_3_wr = [20, 17, 16]

    inds_3_wl_c = [5, 7, 10]
    inds_3_wc_c = [8, 13, 11]
    inds_3_wr_c = [14, 17, 15]

    inds_3_ul = [1, 2, 6, -1]
    inds_3_ur = [18, 19, -2, 20]

    inds_3_ul_c = [3, 4, 7, 8]
    inds_3_ur_c = [12, 19, 13, 14]

    inds_3_ham = [3, 4, 12, 1, 2, 18]
    inds_3_rho = [9, -3, 16, 10, 11, 15]
    env_3 = tn.ncon([
        isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham, rho
    ], [
        inds_3_wl, inds_3_wr, inds_3_wl_c, inds_3_wc_c, inds_3_wr_c, inds_3_ul,
        inds_3_ur, inds_3_ul_c, inds_3_ur_c, inds_3_ham, inds_3_rho
    ])
    return env_3

@tf.contrib.eager.defun
def get_env_isometry_4(ham, rho, isometry, unitary):

    inds_4_wl = [16, 17, 19]
    inds_4_wr = [6, 5, 9]

    inds_4_wl_c = [16, 18, 20]
    inds_4_wc_c = [13, 8, 10]
    inds_4_wr_c = [7, 5, 11]

    inds_4_ul = [14, 15, 17, -1]
    inds_4_ur = [1, 2, -2, 6]

    inds_4_ul_c = [14, 12, 18, 13]
    inds_4_ur_c = [3, 4, 8, 7]

    inds_4_ham = [12, 3, 4, 15, 1, 2]
    inds_4_rho = [19, -3, 9, 20, 10, 11]

    env_4 = tn.ncon([
        isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham, rho
    ], [
        inds_4_wl, inds_4_wr, inds_4_wl_c, inds_4_wc_c, inds_4_wr_c, inds_4_ul,
        inds_4_ur, inds_4_ul_c, inds_4_ur_c, inds_4_ham, inds_4_rho
    ])
    return env_4


@tf.contrib.eager.defun
def get_env_isometry_5(ham, rho, isometry, unitary):

    inds_5_wl = [5, 6, 9]
    inds_5_wc = [18, 19, 20]

    inds_5_wl_c = [5, 7, 10]
    inds_5_wc_c = [8, 13, 11]
    inds_5_wr_c = [14, -2, 15]

    inds_5_ul = [1, 2, 6, 18]
    inds_5_ur = [16, 17, 19, -1]

    inds_5_ul_c = [3, 4, 7, 8]
    inds_5_ur_c = [12, 17, 13, 14]

    inds_5_ham = [3, 4, 12, 1, 2, 16]
    inds_5_rho = [9, 20, -3, 10, 11, 15]

    env_5 = tn.ncon([
        isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham, rho
    ], [
        inds_5_wl, inds_5_wc, inds_5_wl_c, inds_5_wc_c, inds_5_wr_c, inds_5_ul,
        inds_5_ur, inds_5_ul_c, inds_5_ur_c, inds_5_ham, inds_5_rho
    ])
    return env_5

@tf.contrib.eager.defun
def get_env_isometry_6(ham, rho, isometry, unitary):

    inds_6_wl = [12, 13, 15]
    inds_6_wc = [6, 5, 16]

    inds_6_wl_c = [12, 14, 17]
    inds_6_wc_c = [10, 11, 18]
    inds_6_wr_c = [20, -2, 19]

    inds_6_ul = [8, 7, 13, 6]
    inds_6_ur = [1, 2, 5, -1]

    inds_6_ul_c = [8, 9, 14, 10]
    inds_6_ur_c = [3, 4, 11, 20]

    inds_6_ham = [9, 3, 4, 7, 1, 2]
    inds_6_rho = [15, 16, -3, 17, 18, 19]

    env_6 = tn.ncon([
        isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham, rho
    ], [
        inds_6_wl, inds_6_wc, inds_6_wl_c, inds_6_wc_c, inds_6_wr_c, inds_6_ul,
        inds_6_ur, inds_6_ul_c, inds_6_ur_c, inds_6_ham, inds_6_rho
    ])
    return env_6

@tf.contrib.eager.defun
def left_ascending_super_operator(ham, isometry, unitary):
    inds_right_ul = [1, 2, 13, 5]
    inds_right_ur = [6, 9, 7, 16]
    inds_right_ul_c = [3, 4, 14, 10]
    inds_right_ur_c = [8, 9, 11, 17]

    inds_right_iso_l = [12, 13, -4]
    inds_right_iso_c = [5, 7, -5]
    inds_right_iso_r = [16, 15, -6]
    inds_right_iso_l_c = [12, 14, -1]
    inds_right_iso_c_c = [10, 11, -2]
    inds_right_iso_r_c = [17, 15, -3]

    inds_right_ham = [3, 4, 8, 1, 2, 6]

    hright = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_right_iso_l, inds_right_iso_c, inds_right_iso_r,
        inds_right_iso_l_c, inds_right_iso_c_c, inds_right_iso_r_c,
        inds_right_ul, inds_right_ur, inds_right_ul_c, inds_right_ur_c,
        inds_right_ham
    ])
    return hright


@tf.contrib.eager.defun
def right_ascending_super_operator(ham, isometry, unitary):
    inds_left_ul = [8, 6, 13, 7]
    inds_left_ur = [1, 2, 5, 16]
    inds_left_ul_c = [8, 9, 14, 10]
    inds_left_ur_c = [3, 4, 11, 17]
    inds_left_iso_l = [12, 13, -4]
    inds_left_iso_c = [7, 5, -5]
    inds_left_iso_r = [16, 15, -6]
    inds_left_iso_l_c = [12, 14, -1]
    inds_left_iso_c_c = [10, 11, -2]
    inds_left_iso_r_c = [17, 15, -3]
    inds_left_ham = [9, 3, 4, 6, 1, 2]

    hleft = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_left_iso_l, inds_left_iso_c, inds_left_iso_r, inds_left_iso_l_c,
        inds_left_iso_c_c, inds_left_iso_r_c, inds_left_ul, inds_left_ur,
        inds_left_ul_c, inds_left_ur_c, inds_left_ham
    ])
    return hleft



@tf.contrib.eager.defun
def left_descending_super_operator(rho, isometry, unitary):
    inds_left_ul = [-1, -2, 8, 9]
    inds_left_ur = [-3, 12, 10, 11]
    inds_left_ul_c = [-4, -5, 16, 17]
    inds_left_ur_c = [-6, 12, 14, 13]
    inds_left_iso_l = [1, 8, 2]
    inds_left_iso_c = [9, 10, 7]
    inds_left_iso_r = [11, 4, 5]
    inds_left_iso_l_c = [1, 16, 3]
    inds_left_iso_c_c = [17, 14, 15]
    inds_left_iso_r_c = [13, 4, 6]
    inds_left_rho = [2, 7, 5, 3, 15, 6]
    rho = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), rho
    ], [
        inds_left_iso_l, inds_left_iso_c, inds_left_iso_r, inds_left_iso_l_c,
        inds_left_iso_c_c, inds_left_iso_r_c, inds_left_ul, inds_left_ur,
        inds_left_ul_c, inds_left_ur_c, inds_left_rho
    ])
    return rho


@tf.contrib.eager.defun
def right_descending_super_operator(rho, isometry, unitary):
    inds_right_ul = [10, -1, 8, 9]
    inds_right_ur = [-2, -3, 16, 17]
    inds_right_ul_c = [10, -4, 11, 13]
    inds_right_ur_c = [-5, -6, 14, 15]

    inds_right_iso_l = [1, 8, 2]
    inds_right_iso_c = [9, 16, 7]
    inds_right_iso_r = [17, 4, 5]

    inds_right_iso_l_c = [1, 11, 3]
    inds_right_iso_c_c = [13, 14, 12]
    inds_right_iso_r_c = [15, 4, 6]

    inds_right_rho = [2, 7, 5, 3, 12, 6]
    rho = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), rho
    ], [
        inds_right_iso_l, inds_right_iso_c, inds_right_iso_r,
        inds_right_iso_l_c, inds_right_iso_c_c, inds_right_iso_r_c,
        inds_right_ul, inds_right_ur, inds_right_ul_c, inds_right_ur_c,
        inds_right_rho
    ])
    return rho

