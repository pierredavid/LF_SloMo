#Copyright (c) 2020 Pierre David

import OpenEXR
import Imath
import numpy as np
import numpy.ma as ma
import cv2 as cv
import os
import skimage.metrics as sm

def load_sceneflow(filename):
    
    file = OpenEXR.InputFile(filename)
    half = Imath.PixelType(Imath.PixelType.HALF)
    (u, v, d) = [np.frombuffer(file.channel(Chan, half), dtype=np.float16) for Chan in ["R", "G", "B"]]
    w,h = (file.header()['dataWindow'].max.x + 1, file.header()['dataWindow'].max.y + 1)
    u = u.reshape(h,w).astype(np.double)
    v = v.reshape(h,w).astype(np.double)
    d  = d.reshape(h,w).astype(np.double)
    return u, v, d

def load_lightfield(path, Nu, Nv=None, time=-1, uv_inverted=True):
    
    if Nv is None:
        Nv = Nu
    
    im_list = []
    for file in sorted(os.listdir(path)):
        full_path = os.path.join(path,file)
        if os.path.isfile(full_path):
            im_list.append(full_path)  

    if time > 0:
        im_filt = filter(lambda k: f"{time:04}" in os.path.basename(k), im_list)
        im_list = list(im_filt)
    if len(im_list) != Nv*Nu:
        print("Error: the number of found images does not match Nu*Nv")
        return -1
    
    h,w,c = cv.imread(im_list[0]).shape
    dtype = cv.imread(im_list[0]).dtype
    LF = np.zeros(shape=(Nv,Nu,h,w,c), dtype=dtype)
    for v in range(Nv):
        for u in range(Nu):
            LF[v,u,:,:,:] = cv.imread(im_list[u + Nu*v]) if uv_inverted else cv.imread(im_list[v + Nv*u])
        
    return LF

def load_LF_sceneflow(path, Nu, Nv=None, time=-1, uv_inverted=True):
    
    if Nv is None:
        Nv = Nu
    
    sf_list = []
    for file in sorted(os.listdir(path)):
        full_path = os.path.join(path,file)
        _, ext = os.path.splitext(full_path)
        if os.path.isfile(full_path) and ext == ".exr":
            sf_list.append(full_path)
    
    if time > 0:
        sf_filt = filter(lambda k: f"{time:04}" in os.path.basename(k), sf_list)
        sf_list = list(sf_filt)
    if len(sf_list) != Nv*Nu:
        print("Error: the number of found images does not match Nu*Nv")
        return -1
    
    
    dx,_,_ = load_sceneflow(sf_list[0])
    h,w = dx.shape
    dtype = dx.dtype
    LFsf = np.zeros(shape=(Nv,Nu,h,w,3), dtype=dtype)
    for v in range(Nv):
        for u in range(Nu):
            dx,dy,d = load_sceneflow(sf_list[u + Nu*v]) if uv_inverted else load_sceneflow(sf_list[v + Nv*u])
            sf = np.stack((dx,dy,d), axis=-1)
            LFsf[v,u,:,:,:] = sf
        
    return LFsf
    
def compute_warp_std_occ(LF, LF_disp, vh_ratio=1):
    
    Nv, Nu, h, w, chan = LF.shape
    uc = (Nu-1)//2
    vc = (Nv-1)//2
    warp_LF = np.zeros((Nu*Nv,h,w,chan))
    warp_mask = np.zeros((Nu*Nv,h,w,chan), dtype=np.bool)
    disp0 = np.squeeze(np.repeat(LF_disp[vc, uc, :, :, np.newaxis], chan, axis=-1))
    xx, yy, cc = np.meshgrid(range(w), range(h), range(chan))
    idx = 0
    for u in range(Nu):
        for v in range(Nv):
            du = u - uc
            dv = v - vc
            xx0 = np.rint(xx - du*disp0).astype('int')
            yy0 = np.rint(yy - vh_ratio*dv*disp0).astype('int')
            
            mask_x = np.logical_and(xx0 >= 0, xx0 < w)
            mask_y = np.logical_and(yy0 >= 0, yy0 < h)
            mask = np.logical_and(mask_x, mask_y)
            
            xx0[xx0 < 0] = 0
            xx0[xx0 > w-1] = w-1
            yy0[yy0 < 0] = 0
            yy0[yy0 > h-1] = h-1
            subap = np.squeeze(LF[v,u,:,:,:])
            
            disp_uv = np.squeeze(np.repeat(LF_disp[v, u, :, :, np.newaxis], chan, axis=-1))
            warp_disp = disp_uv[yy0, xx0, cc]
            da = max(du, dv)
            eps = 1/da if da != 0 else 0
            mask_d = np.abs(warp_disp - disp0) <= eps
            mask = np.logical_and(mask, mask_d)
            
            warp_LF[idx,:,:,:] = subap[yy0, xx0, cc]
            warp_mask[idx,:,:,:] = mask

            idx += 1
    
    warp_mask = np.invert(warp_mask)
    warp_LF_ma = ma.array(warp_LF, mask=warp_mask)
    warp_std = np.mean(np.var(warp_LF_ma, axis=0), axis=-1)
    
    return warp_std
    
def compute_LFinterp_metrics(LF, LF_gt, disp, vh_ratio=1):
    
    Nv,Nu,_,_,_ = LF.shape
    mean_ssim = 0
    mean_psnr = 0
    for u in range(Nu):
        for v in range(Nv):
            view = np.squeeze(LF[v,u,:,:,:])
            view_gt = np.squeeze(LF_gt[v,u,:,:,:])
            mean_ssim += sm.structural_similarity(view_gt, view, data_range=255, multichannel=True)
            mean_psnr += sm.peak_signal_noise_ratio(view_gt, view, data_range=255)
    mean_psnr /= Nu*Nv
    mean_ssim /= Nu*Nv
    
    warp_std = 1 - np.mean(np.sqrt(compute_warp_std_occ(LF, disp, vh_ratio)))/127.5
    lfec = 10*np.log10(255**2/np.mean(compute_warp_std_occ(LF, disp, vh_ratio)))
    
    return mean_psnr, mean_ssim, warp_std, lfec

def evaluate_method_sintel(root, root_gt, root_sf):
    
    scenes = ["Bamboo2", "Temple1"]
    renders = ["clean", "final"]
    
    Nu = 3
    Nv = 3
    frames = range(2,50)
    
    metrics = np.zeros(shape=(len(frames), len(renders)*len(scenes), 4))
    for i, f in enumerate(frames):
        for s, scene in enumerate(scenes):
            path_sf = os.path.join(root_sf, scene, "scene_flow")
            sf = load_LF_sceneflow(path_sf, Nu, Nv, f)
            disp = np.squeeze(sf[:,:,:,:,2])
            for r, render in enumerate(renders):
                path = os.path.join(root, scene, render)
                path_gt = os.path.join(root_gt, scene, render)

                LF = load_lightfield(path, Nu, Nv, f)
                LF_gt = load_lightfield(path_gt, Nu, Nv, f)
                
                psnr, ssim, ws, wso = compute_LFinterp_metrics(LF, LF_gt, disp)
                metrics[i,r + s*len(renders),0] = psnr
                metrics[i,r + s*len(renders),1] = ssim
                metrics[i,r + s*len(renders),2] = ws
                metrics[i,r + s*len(renders),3] = wso
#        advancement = int(100*i/len(frames))
#        print(advancement)
                
    return metrics

if __name__ == "__main__":
    
    root_gt = "/nfs/nas4/sirocco_clim/sirocco_clim_image/Synthetic/Sintel/Rendu/Sparse"
    root_sf = "/nfs/nas4/sirocco_clim/sirocco_clim_image/Synthetic/Sintel/Rendu/Sparse"
    
    root_interp = "/home/pdavid/Documents/Results/igrida/ICME/SuperSloMo"
    metrics = evaluate_method_sintel(root_interp, root_gt, root_sf)
    print(np.mean(metrics_slomo, axis=0))
    np.save("metrics.npy", metrics)

