import numpy as np
import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
import skimage
import time
import os

# System PATH for VC++ Compiler (cl.exe)
if (os.system("cl.exe")):
    os.environ['PATH'] += ';'+r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.32.31326\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

M_PI=3.14159265358979323846

def build_list_3d_sphere(radius):
    ar_len=0
    x_list=[]
    y_list=[]
    z_list=[]
    unit_x_list=[]
    unit_y_list=[]
    unit_z_list=[]
    theta_list=[]
    phi_list=[]
    radius_list=[]
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            for k in range(-radius,radius+1):
                r=np.sqrt(i*i+j*j+k*k)
                if ((r < radius+1.0) and not (i==0 and j==0 and k==0)):
                    x_list.append(i)
                    y_list.append(j)
                    z_list.append(k)
                    
                    unit_x_list.append(i/r)
                    unit_y_list.append(j/r)
                    unit_z_list.append(k/r)
                    
                    theta=math.atan2(i,j)
                    if theta<0.0:
                        theta+=M_PI*2
                    theta_list.append(theta)
                    
                    phi=math.atan2(k,np.sqrt(i*i+j*j))
                    if phi<0.0:
                        phi+=M_PI*2
                    phi_list.append(phi)
                    
                    radius_list.append(r)
    zipped=zip(x_list, y_list, z_list, unit_x_list, unit_y_list, unit_z_list, phi_list, theta_list, radius_list)
    #          0       1       2       3            4            5            6         7           8
    zipped=sorted(zipped, key = lambda x: (x[6], x[7], x[8]))
    #zipped=zip(x_list, y_list, deg_list, radius_list)
    #zipped=sorted(zipped, key = lambda x: (x[2], x[3]))
    return zipped

def generate_surface_dp_list(ar_list, radius):
    x_list, y_list, z_list, unit_x_list, unit_y_list, unit_z_list, phi_list, theta_list, radius_list = zip(*ar_list)
    
    # find sphere surface index in the list
    surface_indexs=[]
    for index in range(len(radius_list)):
        min_radius=radius-0.0001
        if radius_list[index]>min_radius:
            surface_indexs.append(index)
    
    hemisphere_indexs=[] #len should equalt to len(surface_indexs)
    for idx in range(len(surface_indexs)):
        center_index=surface_indexs[idx]
        center_unit_x=unit_x_list[center_index]
        center_unit_y=unit_y_list[center_index]
        center_unit_z=unit_z_list[center_index]
        positive_hemisphere=[]
        negative_hemisphere=[]
        for index in range(len(radius_list)):
            if center_index == index:
                continue
            target_unit_x=unit_x_list[index]
            target_unit_y=unit_y_list[index]
            target_unit_z=unit_z_list[index]
            target_dot_prdouct=(center_unit_x*target_unit_x)+(center_unit_y*target_unit_y)+(center_unit_z*target_unit_z)
            if target_dot_prdouct>0.0001:
                positive_hemisphere.append(index)
            elif target_dot_prdouct<-0.0001:
                negative_hemisphere.append(index)
        hemisphere_indexs.append([positive_hemisphere, negative_hemisphere])
    
    #print(len(hemisphere_indexs))
    
    hemisphere_dp_pos_add=[]
    hemisphere_dp_pos_sub=[]
    hemisphere_dp_neg_add=[]
    hemisphere_dp_neg_sub=[]
    
    # First add all
    hemisphere_dp_pos_add.append(hemisphere_indexs[0][0])
    hemisphere_dp_pos_sub.append([])
    hemisphere_dp_neg_add.append(hemisphere_indexs[0][1])
    hemisphere_dp_neg_sub.append([])
    
    prev_pos=hemisphere_indexs[0][0]
    prev_neg=hemisphere_indexs[0][1]
    for idx in range(1,len(surface_indexs)):
        current_pos=hemisphere_indexs[idx][0]
        current_neg=hemisphere_indexs[idx][1]
        pos_add=list(set(current_pos)-set(prev_pos))
        pos_sub=list(set(prev_pos)-set(current_pos))
        neg_add=list(set(current_neg)-set(prev_neg))
        neg_sub=list(set(prev_neg)-set(current_neg))
        hemisphere_dp_pos_add.append(pos_add)
        hemisphere_dp_pos_sub.append(pos_sub)
        hemisphere_dp_neg_add.append(neg_add)
        hemisphere_dp_neg_sub.append(neg_sub)
        prev_pos=current_pos
        prev_neg=current_neg
    
    
    dp_pos_add_start_idx=0
    dp_pos_add_start_idxs=[]
    dp_pos_add_start_lens=[]
    
    dp_pos_sub_start_idx=0
    dp_pos_sub_start_idxs=[]
    dp_pos_sub_start_lens=[]
    
    dp_neg_add_start_idx=0
    dp_neg_add_start_idxs=[]
    dp_neg_add_start_lens=[]
    
    dp_neg_sub_start_idx=0
    dp_neg_sub_start_idxs=[]
    dp_neg_sub_start_lens=[]

    for idx in range(0, len(surface_indexs)):
        dp_pos_add_start_idxs.append(dp_pos_add_start_idx)
        dp_pos_sub_start_idxs.append(dp_pos_sub_start_idx)
        dp_neg_add_start_idxs.append(dp_neg_add_start_idx)
        dp_neg_sub_start_idxs.append(dp_neg_sub_start_idx)
        
        dp_pos_add_start_len=len(hemisphere_dp_pos_add[idx])
        dp_pos_add_start_lens.append(dp_pos_add_start_len)
        dp_pos_add_start_idx+=dp_pos_add_start_len
        
        dp_pos_sub_start_len=len(hemisphere_dp_pos_sub[idx])
        dp_pos_sub_start_lens.append(dp_pos_sub_start_len)
        dp_pos_sub_start_idx+=dp_pos_sub_start_len
        
        dp_neg_add_start_len=len(hemisphere_dp_neg_add[idx])
        dp_neg_add_start_lens.append(dp_neg_add_start_len)
        dp_neg_add_start_idx+=dp_neg_add_start_len
        
        dp_neg_sub_start_len=len(hemisphere_dp_neg_sub[idx])
        dp_neg_sub_start_lens.append(dp_neg_sub_start_len)
        dp_neg_sub_start_idx+=dp_neg_sub_start_len
    
    hemisphere_dp_pos_add=sum(hemisphere_dp_pos_add, [])
    hemisphere_dp_pos_sub=sum(hemisphere_dp_pos_sub, [])
    hemisphere_dp_neg_add=sum(hemisphere_dp_neg_add, [])
    hemisphere_dp_neg_sub=sum(hemisphere_dp_neg_sub, [])

    return surface_indexs, \
           hemisphere_dp_pos_add, hemisphere_dp_pos_sub, hemisphere_dp_neg_add, hemisphere_dp_neg_sub, \
           dp_pos_add_start_idxs, dp_pos_sub_start_idxs, dp_neg_add_start_idxs, dp_neg_sub_start_idxs, \
           dp_pos_add_start_lens, dp_pos_sub_start_lens, dp_neg_add_start_lens, dp_neg_sub_start_lens


#Initial PyCUDA
mod = SourceModule(open('kernel_3d.cu').read())

def SFEGO_3D(np_input_data, dim_x, dim_y, dim_z, radius):
    ar_list = build_list_3d_sphere(radius)
    x_list, y_list, z_list, unit_x_list, unit_y_list, unit_z_list, phi_list, theta_list, radius_list = zip(*ar_list)
    list_len=len(x_list)
    
    surface_indexs, \
    hemisphere_dp_pos_add, hemisphere_dp_pos_sub, hemisphere_dp_neg_add, hemisphere_dp_neg_sub, \
    dp_pos_add_start_idxs, dp_pos_sub_start_idxs, dp_neg_add_start_idxs, dp_neg_sub_start_idxs, \
    dp_pos_add_start_lens, dp_pos_sub_start_lens, dp_neg_add_start_lens, dp_neg_sub_start_lens = generate_surface_dp_list(ar_list, radius)
    dp_len=len(surface_indexs)
    
    np_x_list = np.asarray(x_list).astype(np.int32)
    np_y_list = np.asarray(y_list).astype(np.int32)
    np_z_list = np.asarray(z_list).astype(np.int32)
    np_unit_x_list = np.asarray(unit_x_list).astype(np.float32)
    np_unit_y_list = np.asarray(unit_y_list).astype(np.float32)
    np_unit_z_list = np.asarray(unit_z_list).astype(np.float32)
    np_surface_indexs = np.asarray(surface_indexs).astype(np.int32)
    np_hemisphere_dp_pos_add = np.asarray(hemisphere_dp_pos_add).astype(np.int32)
    np_hemisphere_dp_pos_sub = np.asarray(hemisphere_dp_pos_sub).astype(np.int32)
    np_hemisphere_dp_neg_add = np.asarray(hemisphere_dp_neg_add).astype(np.int32)
    np_hemisphere_dp_neg_sub = np.asarray(hemisphere_dp_neg_sub).astype(np.int32)
    np_dp_pos_add_start_idxs = np.asarray(dp_pos_add_start_idxs).astype(np.int32)
    np_dp_pos_sub_start_idxs = np.asarray(dp_pos_sub_start_idxs).astype(np.int32)
    np_dp_neg_add_start_idxs = np.asarray(dp_neg_add_start_idxs).astype(np.int32)
    np_dp_neg_sub_start_idxs = np.asarray(dp_neg_sub_start_idxs).astype(np.int32)
    np_dp_pos_add_start_lens = np.asarray(dp_pos_add_start_lens).astype(np.int32)
    np_dp_pos_sub_start_lens = np.asarray(dp_pos_sub_start_lens).astype(np.int32)
    np_dp_neg_add_start_lens = np.asarray(dp_neg_add_start_lens).astype(np.int32)
    np_dp_neg_sub_start_lens = np.asarray(dp_neg_sub_start_lens).astype(np.int32)
    
    data = cuda.mem_alloc(np_input_data.nbytes)
    cuda.memcpy_htod(data, np_input_data)
    diff = cuda.mem_alloc(np_input_data.nbytes)
    direct_x = cuda.mem_alloc(np_input_data.nbytes)
    direct_y = cuda.mem_alloc(np_input_data.nbytes)
    direct_z = cuda.mem_alloc(np_input_data.nbytes)
    result = cuda.mem_alloc(np_input_data.nbytes)
    
    list_x = cuda.mem_alloc(np_x_list.nbytes)
    cuda.memcpy_htod(list_x, np_x_list)
    list_y = cuda.mem_alloc(np_y_list.nbytes)
    cuda.memcpy_htod(list_y, np_y_list)
    list_z = cuda.mem_alloc(np_z_list.nbytes)
    cuda.memcpy_htod(list_z, np_z_list)
    list_unit_x = cuda.mem_alloc(np_unit_x_list.nbytes)
    cuda.memcpy_htod(list_unit_x, np_unit_x_list)
    list_unit_y = cuda.mem_alloc(np_unit_y_list.nbytes)
    cuda.memcpy_htod(list_unit_y, np_unit_y_list)
    list_unit_z = cuda.mem_alloc(np_unit_z_list.nbytes)
    cuda.memcpy_htod(list_unit_z, np_unit_z_list)
    
    cu_surface_indexs = cuda.mem_alloc(np_surface_indexs.nbytes)
    cuda.memcpy_htod(cu_surface_indexs, np_surface_indexs)
    cu_hemisphere_dp_pos_add = cuda.mem_alloc(np_hemisphere_dp_pos_add.nbytes)
    cuda.memcpy_htod(cu_hemisphere_dp_pos_add, np_hemisphere_dp_pos_add)
    cu_hemisphere_dp_pos_sub = cuda.mem_alloc(np_hemisphere_dp_pos_sub.nbytes)
    cuda.memcpy_htod(cu_hemisphere_dp_pos_sub, np_hemisphere_dp_pos_sub)
    cu_hemisphere_dp_neg_add = cuda.mem_alloc(np_hemisphere_dp_neg_add.nbytes)
    cuda.memcpy_htod(cu_hemisphere_dp_neg_add, np_hemisphere_dp_neg_add)
    cu_hemisphere_dp_neg_sub = cuda.mem_alloc(np_hemisphere_dp_neg_sub.nbytes)
    cuda.memcpy_htod(cu_hemisphere_dp_neg_sub, np_hemisphere_dp_neg_sub)

    cu_dp_pos_add_start_idxs = cuda.mem_alloc(np_dp_pos_add_start_idxs.nbytes)
    cuda.memcpy_htod(cu_dp_pos_add_start_idxs, np_dp_pos_add_start_idxs)
    cu_dp_pos_sub_start_idxs = cuda.mem_alloc(np_dp_pos_sub_start_idxs.nbytes)
    cuda.memcpy_htod(cu_dp_pos_sub_start_idxs, np_dp_pos_sub_start_idxs)
    cu_dp_neg_add_start_idxs = cuda.mem_alloc(np_dp_neg_add_start_idxs.nbytes)
    cuda.memcpy_htod(cu_dp_neg_add_start_idxs, np_dp_neg_add_start_idxs)
    cu_dp_neg_sub_start_idxs = cuda.mem_alloc(np_dp_neg_sub_start_idxs.nbytes)
    cuda.memcpy_htod(cu_dp_neg_sub_start_idxs, np_dp_neg_sub_start_idxs)
    
    cu_dp_pos_add_start_lens = cuda.mem_alloc(np_dp_pos_add_start_lens.nbytes)
    cuda.memcpy_htod(cu_dp_pos_add_start_lens, np_dp_pos_add_start_lens)
    cu_dp_pos_sub_start_lens = cuda.mem_alloc(np_dp_pos_sub_start_lens.nbytes)
    cuda.memcpy_htod(cu_dp_pos_sub_start_lens, np_dp_pos_sub_start_lens)
    cu_dp_neg_add_start_lens = cuda.mem_alloc(np_dp_neg_add_start_lens.nbytes)
    cuda.memcpy_htod(cu_dp_neg_add_start_lens, np_dp_neg_add_start_lens)
    cu_dp_neg_sub_start_lens = cuda.mem_alloc(np_dp_neg_sub_start_lens.nbytes)
    cuda.memcpy_htod(cu_dp_neg_sub_start_lens, np_dp_neg_sub_start_lens)

    #Define CUDA Function
    knl_gradient_fnc = mod.get_function("SFEGO_3d_gradient")
    knl_integral_fnc = mod.get_function("SFEGO_3d_integral")
    
    #Calculate CUDA Execution Dimension
    bdim = (8, 8, 8)
    dx, mx = divmod(dim_x, bdim[0])
    dy, my = divmod(dim_y, bdim[1])
    dz, mz = divmod(dim_z, bdim[2])
    gdim = ( (dx + (mx>0)), (dy + (my>0)), (dz + (mz>0)) )
    
    #CUDA Execution
    knl_gradient_fnc(data, diff, direct_x, direct_y, direct_z, \
                 list_x, list_y, list_z, list_unit_x, list_unit_y, list_unit_z, \
                 cu_surface_indexs, \
                 cu_hemisphere_dp_pos_add, cu_hemisphere_dp_pos_sub, \
                 cu_hemisphere_dp_neg_add, cu_hemisphere_dp_neg_sub, \
                 cu_dp_pos_add_start_idxs, cu_dp_pos_sub_start_idxs, \
                 cu_dp_neg_add_start_idxs, cu_dp_neg_sub_start_idxs, \
                 cu_dp_pos_add_start_lens, cu_dp_pos_sub_start_lens, \
                 cu_dp_neg_add_start_lens, cu_dp_neg_sub_start_lens, \
                 np.int32(list_len), np.int32(dp_len), \
                 np.int32(dim_x), np.int32(dim_y), np.int32(dim_z), \
                 block=bdim, grid=gdim)
    knl_integral_fnc(result, diff, direct_x, direct_y, direct_z, \
                 list_x, list_y, list_z, list_unit_x, list_unit_y, list_unit_z, \
                 np.int32(list_len), np.int32(dim_x), np.int32(dim_y), np.int32(dim_z), \
                 block=bdim, grid=gdim)

    #Get CUDA Result
    np_result = np.empty_like(np_input_data)
    cuda.memcpy_dtoh(np_result, result)
    np_result = np_result / list_len
    
    data.free()
    diff.free()
    direct_x.free()
    direct_y.free()
    direct_z.free()
    result.free()
    list_x.free()
    list_y.free()
    list_z.free()
    list_unit_x.free()
    list_unit_y.free()
    list_unit_z.free()
    cu_surface_indexs.free()
    cu_hemisphere_dp_pos_add.free()
    cu_hemisphere_dp_pos_sub.free()
    cu_hemisphere_dp_neg_add.free()
    cu_hemisphere_dp_neg_sub.free()
    cu_dp_pos_add_start_idxs.free()
    cu_dp_pos_sub_start_idxs.free()
    cu_dp_neg_add_start_idxs.free()
    cu_dp_neg_sub_start_idxs.free()
    cu_dp_pos_add_start_lens.free()
    cu_dp_pos_sub_start_lens.free()
    cu_dp_neg_add_start_lens.free()
    cu_dp_neg_sub_start_lens.free()
    
    return np_result

def GenereateSimulationData():
    dim_x=128
    dim_y=129
    dim_z=130
    np_data=np.zeros(dim_x*dim_y*dim_z).astype(np.float32)
    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                cx=64-x
                cy=64-y
                cz=64-z
                radius=np.sqrt(cx*cx+cy*cy+cz*cz)
                np_data[z*dim_y*dim_x+y*dim_x+x]+=math.sin(radius/3.0)
                np_data[z*dim_y*dim_x+y*dim_x+x]+=math.sin(x)+math.sin(y)+math.sin(z)
        
    return np_data, dim_x, dim_y, dim_z

print("Generate Simulation Data...")
np_data, dim_x, dim_y, dim_z=GenereateSimulationData()
print("Done!!\n")

start_time = time.time()
file = open('default_radius')
for line in file:
    fields = line.strip().split()
    resize_ratio=float(fields[0])
    execute_radius=int(fields[1])
    target_dim_x=int(dim_x/resize_ratio)
    target_dim_y=int(dim_y/resize_ratio)
    target_dim_z=int(dim_z/resize_ratio)
    
    print("resize_ratio="+str(resize_ratio)+" execute_radius="+str(execute_radius)+" effective_radius="+str(resize_ratio*execute_radius)+" size:(z,y,x)=("+str(target_dim_z)+", "+str(target_dim_y)+", "+str(target_dim_x)+")")
    np_3d_data = np_data.reshape((dim_z, dim_y, dim_x))
    np_3d_input_data = skimage.transform.resize(np_3d_data, (target_dim_z, target_dim_y, target_dim_x))
    np_input_data = np_3d_input_data.flatten()
    
    np_result=SFEGO_3D(np_input_data, target_dim_x, target_dim_y, target_dim_z, execute_radius)
    
    np_3d_result = np_result.reshape((target_dim_z, target_dim_y, target_dim_x))
    np_3d_output_result = skimage.transform.resize(np_3d_result, (dim_z, dim_y, dim_x))
    print("Done!!\n")
    
    for z in range(dim_z):
        np_2d_input = np_3d_data[z].copy()
        np_2d_result = np_3d_output_result[z].copy()
        
        result_min=np.min(np_2d_result)
        result_max=np.max(np_2d_result)
        np_2d_result = (255*(np_2d_result-result_min)/(result_max-result_min)).astype(np.uint8)
        
        input_min=np.min(np_2d_input)
        input_max=np.max(np_2d_input)
        np_2d_input = (255*(np_2d_input-input_min)/(input_max-input_min)).astype(np.uint8)
        
        output = cv2.hconcat([np_2d_input, np_2d_result])
        cv2.imshow('Input v.s. Output', output)
        cv2.waitKey(16)

end_time=time.time()
used_time=end_time-start_time
print("Used Time:", used_time)