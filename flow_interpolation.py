#Copyright (c) 2020 Pierre David

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2 as cv

def oneway_flow_interp(F_01, im0, im1, time):
    h, w, _ = F_01.shape
    xx, yy = np.meshgrid(range(w), range(h))
    nxx = xx + time*F_01[:,:,0]
    nyy = yy + time*F_01[:,:,1]
    
    rxx = np.rint(xx + F_01[:,:,0])
    rxx = rxx.astype("int")
    ryy = np.rint(yy + F_01[:,:,1])
    ryy = ryy.astype("int")
    
    mask = np.all([rxx < rxx.shape[1], rxx >= 0, ryy < ryy.shape[0], ryy >= 0], axis=0)
    mask = np.invert(mask).flatten()
    
    rxx[rxx < 0] = 0
    rxx[rxx > w-1] = w-1
    rxx = rxx.flatten()
    
    ryy[ryy < 0] = 0
    ryy[ryy > h-1] = h-1
    ryy = ryy.flatten()   
    
    rxy = np.ravel_multi_index((ryy,rxx), (h,w))
    
    im0 = im0.reshape((h*w,3))
    im1 = im1.reshape((h*w,3))
    im1_warped = im1[rxy,:]
    
    dist = np.sum((im0 - im1_warped)**2, axis=1)
    dist[mask] = 1e10
    
    cxx_1 = np.floor(nxx)
    cxx_1 = cxx_1.astype("int")
    cxx_2 = cxx_1 + 1
    cxx_1[cxx_1 < 0] = 0
    cxx_1[cxx_1 > w-1] = w-1
    cxx_2[cxx_2 < 0] = 0
    cxx_2[cxx_2 > w-1] = w-1
    
    cyy_1 = np.floor(nyy)
    cyy_1 = cyy_1.astype("int")
    cyy_2 = cyy_1 + 1
    cyy_1[cyy_1 < 0] = 0
    cyy_1[cyy_1 > h-1] = h-1
    cyy_2[cyy_2 < 0] = 0
    cyy_2[cyy_2 > h-1] = h-1
    
    idx11 = np.ravel_multi_index((cyy_1,cxx_1), (h,w)).flatten()
    idx12 = np.ravel_multi_index((cyy_1,cxx_2), (h,w)).flatten()
    idx21 = np.ravel_multi_index((cyy_2,cxx_1), (h,w)).flatten()
    idx22 = np.ravel_multi_index((cyy_2,cxx_2), (h,w)).flatten()
    
    idx = np.concatenate((idx11, idx12, idx21, idx22), axis=0)
    
    uf = F_01[:,:,0].flatten()
    vf = F_01[:,:,1].flatten()
    
    k = np.arange(h*w, dtype=int)
    k = np.tile(k, 4)
    
    index, unique_idx, count = np.unique(idx11, return_index=True, return_counts=True)
    mask = count != 1
    
    k_unique = k[unique_idx[~mask]]
    k_corres = index[~mask]
    
    zu = np.full((h*w,), 10**9)
    zv = np.full((h*w,), 10**9)
    
    zu[k_corres] = uf[k_unique]
    zv[k_corres] = vf[k_unique]

    for i in index[mask]:
        idx_0 = k[idx == i]
        score = dist[idx_0]
        n_min = np.argmin(score)
        zu[i] = uf[idx_0[n_min]]
        zv[i] = vf[idx_0[n_min]]
    
    zu = zu.reshape((h,w))
    zv = zv.reshape((h,w))
    
    return zu, zv

def outsidein_interp(u_t, v_t):
    
    h,w = u_t.shape
    
    mask = u_t > 10**8
    N_empty = np.sum(mask)
    while N_empty > 0:
        mask_c = (255*mask).astype(np.uint8)
        contours, _ = cv.findContours(mask_c, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for c in contours:      
            cont = np.squeeze(c, axis=1)
            cy = cont[:,1]
            cx = cont[:,0]
            nx = np.stack((cx-1, cx-1, cx-1,   cx,   cx, cx+1, cx+1, cx+1), axis=-1)
            ny = np.stack((cy-1,   cy, cy+1, cy-1, cy+1, cy-1,   cy, cy+1), axis=-1)
            
            mask_in = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
            nx[~mask_in] = 0
            ny[~mask_in] = 0
            mask_done = ~mask[ny,nx]
            mask_done[~mask_in] = False   
            
            u_neigh = u_t[ny,nx]
            v_neigh = v_t[ny,nx]
            
            total = np.sum(mask_done, axis=1)
            total[total == 0] = 1
            
            u_cont = np.sum(mask_done*u_neigh, axis=1)/total
            v_cont = np.sum(mask_done*v_neigh, axis=1)/total
            
            u_t[cy,cx] = u_cont
            v_t[cy,cx] = v_cont
        
        mask = u_t > 10**8
        N_empty = np.sum(mask)

    return u_t, v_t

def flow_interp(F_01, F_10, im0, im1, time):
    
    u_t1, v_t1 = oneway_flow_interp(F_01, im0, im1, time)   
    u_t0, v_t0 = oneway_flow_interp(F_10, im1, im0, 1 - time)

#    u_t0, v_t0 = outsidein_interp(u_t0, v_t0)
#    u_t1, v_t1 = outsidein_interp(u_t1, v_t1)
#    u_t = (1-time)*u_t1 - time*u_t0
#    v_t = (1-time)*v_t1 - time*v_t0

    m_t1 = (1-time)*(u_t1 < 10**8)
    m_t0 = (time)*(u_t0 < 10**8)
    mask = (u_t1 < 10**8) | (u_t0 < 10**8)
    
    m_tot = m_t1 + m_t0
    m_tot[~mask] = 1.
    
    m_t1 = m_t1/m_tot
    m_t0 = m_t0/m_tot
    
    u_t = m_t1*u_t1 - m_t0*u_t0
    v_t = m_t1*v_t1 - m_t0*v_t0
    
    u_t[~mask] = 10**9
    v_t[~mask] = 10**9

    u_t, v_t = outsidein_interp(u_t, v_t)
    
    u_t = gaussian_filter(u_t, sigma=1)
    v_t = gaussian_filter(v_t, sigma=1)
    
    return u_t, v_t

