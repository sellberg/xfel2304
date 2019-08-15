import sys
import math
import time
import struct
import numpy as np
import scipy as sp
from scipy import ndimage
from collections import OrderedDict
import multiprocessing as mp
import ctypes
import helper

Pi = math.pi
    
# angular cross-correlation analysis stuff
#
class correlation_analysis:    
    
    def __init__(self, dir_save, numproc, chunksz, qrange, phirange, n_phi, n_max, q1_list, interp_order, det_sam, wavelng, pixsz, binfactor, dpcenter, dpcentertype,\
                xcca_diffimages, xcca_pairs_type, xcca_intra, xcca_inter, xcca_intra_inter, xcca_qq, xcca_q1q2, xcca_out_ccf, xcca_out_fc_ampl, xcca_out_fc, verbose):
        
        
        # parameters adjustment related to data binning
        self.binfactor=int(binfactor)
        if binfactor>1: 
            qrange[0]=int(qrange[0]/binfactor)
            qrange[1]=int(qrange[1]/binfactor)
            dpcenter[0]=dpcenter[0]/float(binfactor)
            dpcenter[1]=dpcenter[1]/float(binfactor)
            q1_list[:]= [int(x/binfactor) for x in q1_list]
            pixsz*=binfactor
        
        # parameters check and adjustments
        n_max+=1 # to accomodate FC with n=50, one needs to slice at [:, 51]
        pixsz*=0.001; # conversion to [mm]
        dpcenter[0], dpcenter[1] = dpcenter[1], dpcenter[0] # exchange (cenx, ceny)[see e.g. coordinates system of ImageJ] to be compatible with python array indexing 
        
        if xcca_intra_inter: #   'intra-inter' is a difference 'xcca_intra'-'xcca_inter'
            xcca_intra=True
            xcca_inter=True 
    
        if xcca_diffimages:  # calculated separately
            xcca_intra_inter=False
            xcca_intra=False
            xcca_inter=False
                   
        self.q_min=qrange[0]
        self.q_max=qrange[1]
        self.n_q=int((qrange[1]-qrange[0])/qrange[2])    # radial 'q' sampling of intensity
        self.phi_min=phirange[0]
        self.phi_max=phirange[1]
        self.n_phi=n_phi                                 # angular samplig of intensity [points], the same for all resolution rings
        self.n_max=n_max
        self.wavelng=wavelng
        self.dz=det_sam
        self.pixsz=pixsz
        self.dpcenter=dpcenter
        self.dpcentertype=dpcentertype
        self.interp_order=interp_order               
        self.dirsave=dir_save
        self.numproc=numproc
        self.chunksize=chunksz
        self.xcca_diffimages=xcca_diffimages
        self.pairs_type=xcca_pairs_type
        self.xcca_intra=xcca_intra                 
        self.xcca_inter=xcca_inter               
        self.xcca_intra_inter=xcca_intra_inter     
        self.xcca_qq=xcca_qq                         
        self.xcca_q1q2=xcca_q1q2              
        self.xcca_out_ccf=xcca_out_ccf            
        self.xcca_out_fc_ampl=xcca_out_fc_ampl    
        self.xcca_out_fc=xcca_out_fc  
        self.verbose=verbose
        
        q1_list=list(OrderedDict.fromkeys(q1_list)) # remove dublicates and sort      
        if len(q1_list) > 0: 
            q1_list=np.asarray(q1_list)
            min1=q1_list.min(); max1=q1_list.max()
            if (min1<0) | (max1>=self.n_q): 
                print('specified list q1_list={}'.format(q1_list)) 
                print('Error: some values in q1_list fall outside of the range of q values considered in the analysis ({}, {})'.format(0,self.n_q-1))
                sys.exit(1)
        else:
            q1_list=np.arange(self.n_q)                   
        self.q1_list=q1_list        
        self.n_q1=self.q1_list.size
        print('q1_list={}'.format(self.q1_list))
        
        # arrays of polar coordinates q and phi
        self.pol_q = np.outer( np.arange(self.n_q)*(self.q_max-self.q_min)/float(self.n_q) + self.q_min, np.ones(self.n_phi) )
        self.pol_phi = np.outer( np.ones(self.n_q), np.arange(self.n_phi)*(self.phi_max-self.phi_min)/float(self.n_phi) + self.phi_min)
        
        # cartesian coordinates
        self.cart_x_orig = self.pol_q*np.cos(self.pol_phi)
        self.cart_y_orig = self.pol_q*np.sin(self.pol_phi)
        
        # in the case of fixed dp_center position for all images, one can readily define the absolute coordinates in Cartesian system    
        if self.dpcentertype==0: 
            self.cart_x = self.cart_x_orig + self.dpcenter[0]
            self.cart_y = self.cart_y_orig + self.dpcenter[1]
        
        self.theta=np.arctan2((np.arange(self.n_q)*(self.q_max-self.q_min)/float(self.n_q) + self.q_min)*self.pixsz, self.dz) # scattering angle for each q-ring
        self.qvals=4.0*Pi/self.wavelng*np.sin(self.theta*0.5) # |q|-value for each q-ring
        
        self._allocate_data_arrays()
        
        self.cnt_all=0  # number of all processed images
        self.cnt_good=0 # number of good images used in xcca
        self.cnt_pairs=0 # number of valid image pairs used in case of self.xcca_diffimages==True
        self.chunk_numpairs=0 # number of image pairs in the current chunk
    
    # memory allocation    
    def _allocate_data_arrays(self):
        
        self.img_data_pol_shape=(self.chunksize, self.n_q, self.n_phi)
        self.img_data_pol_size=int(self.img_data_pol_shape[0]*self.img_data_pol_shape[1]*self.img_data_pol_shape[2])
        self.img_data_pol=np.frombuffer(mp.Array(ctypes.c_double, self.img_data_pol_size).get_obj()).reshape(self.img_data_pol_shape)              # scattering data, assembled image
        self.mask_data_pol=np.frombuffer(mp.Array(ctypes.c_double, self.img_data_pol_size).get_obj()).reshape(self.img_data_pol_shape)      # scattering data, assembled image
            
        self.iav_average=np.zeros(self.n_q)
        self.iav_average_shared=np.frombuffer(mp.Array(ctypes.c_double, self.numproc*self.n_q).get_obj()).reshape(self.numproc,self.n_q)
        
        if self.xcca_diffimages:
            if self.xcca_qq: 
                self.diff_intra_ccf_qq_average=np.zeros((self.n_q, self.n_phi))
                self.diff_intra_ccf_qq_average_shared=np.frombuffer(mp.Array(ctypes.c_double, self.numproc*self.n_q*self.n_phi).get_obj()).reshape(self.numproc, self.n_q, self.n_phi)
            if self.xcca_q1q2: 
                self.diff_intra_ccf_q1q2_average=np.zeros((self.n_q1, self.n_q, self.n_phi))
                self.diff_intra_ccf_q1q2_average_shared=np.frombuffer(mp.Array(ctypes.c_double, self.numproc*self.n_q1*self.n_q*self.n_phi).get_obj()).reshape(self.numproc, self.n_q1, self.n_q, self.n_phi)
            
        if self.xcca_intra:
            if self.xcca_qq: 
                self.intra_ccf_qq_average=np.zeros((self.n_q, self.n_phi))
                self.intra_ccf_qq_average_shared=np.frombuffer(mp.Array(ctypes.c_double, self.numproc*self.n_q*self.n_phi).get_obj()).reshape(self.numproc, self.n_q, self.n_phi)
            if self.xcca_q1q2: 
                self.intra_ccf_q1q2_average=np.zeros((self.n_q1, self.n_q, self.n_phi))
                self.intra_ccf_q1q2_average_shared=np.frombuffer(mp.Array(ctypes.c_double, self.numproc*self.n_q1*self.n_q*self.n_phi).get_obj()).reshape(self.numproc, self.n_q1, self.n_q, self.n_phi)
            
        if self.xcca_inter:
            if self.xcca_qq: 
                self.inter_ccf_qq_average=np.zeros((self.n_q, self.n_phi))
                self.inter_ccf_qq_average_shared=np.frombuffer(mp.Array(ctypes.c_double, self.numproc*self.n_q*self.n_phi).get_obj()).reshape(self.numproc, self.n_q, self.n_phi)
            if self.xcca_q1q2: 
                self.inter_ccf_q1q2_average=np.zeros((self.n_q1, self.n_q, self.n_phi))       
                self.inter_ccf_q1q2_average_shared=np.frombuffer(mp.Array(ctypes.c_double, self.numproc*self.n_q1*self.n_q*self.n_phi).get_obj()).reshape(self.numproc, self.n_q1, self.n_q, self.n_phi)
                
        if self.xcca_intra_inter:
            if self.xcca_qq: 
                self.intra_inter_ccf_qq_average=np.zeros((self.n_q, self.n_phi))
            if self.xcca_q1q2: 
                self.intra_inter_ccf_q1q2_average=np.zeros((self.n_q1, self.n_q, self.n_phi))
    
    # perform xcca of the data chunk
    def data_chunk_analysis(self, imgnum, data, mask, ids, isgood):
        
        #self.start_time = time.time()
        self.chunk_size_current=imgnum
        self.ids=ids 
        self.isgood=isgood
        
        if self.xcca_inter | self.xcca_intra_inter | self.xcca_diffimages: # find suitable pairs of images
            self._find_image_pairs()
        
        #eetime = time.time(); print('finding image pairs took {:.4f} s'.format(eetime-self.start_time)) 
        
        if self.xcca_diffimages:  # form difference images 
            if self.chunk_numpairs!=0: 
                mp_wrapper(self.numproc, self._imgdiff_transform_to_polar_worker, data, mask, self.dpcenter)   
        else:  
            mp_wrapper(self.numproc, self._img_transform_to_polar_worker, imgnum, data, mask, self.dpcenter)     
        
        #eetime = time.time(); print('transforming to polar coordinates took {:.4f} s'.format(eetime-self.start_time)) 
        
        # accumulate azimuthally averaged intensity for difference images or diffraction patterns, depending on the input options      
        mp_wrapper(self.numproc, self._intensity_average_azimuthal_worker)
        
        #eetime = time.time(); print('azimuthal intensity integration took {:.4f} s'.format(eetime-self.start_time)) 
        
        # compute difference image intra-ccfs
        if self.xcca_diffimages & (self.chunk_numpairs!=0):
            mp_wrapper(self.numproc, self._ccf_diff_intra_worker, self.chunk_numpairs)
        
        #eetime = time.time(); print('difference image intra-ccfs calculation took {:.4f} s'.format(eetime-self.start_time))
              
        # compute intra-ccfs
        if self.xcca_intra:
            mp_wrapper(self.numproc, self._ccf_intra_worker, self.chunk_size_current)
        
        #eetime = time.time(); print('intra-ccfs calculation took {:.4f} s'.format(eetime-self.start_time))
        
        # compute inter-ccfs
        if (self.xcca_inter | self.xcca_intra_inter) & (self.chunk_numpairs!=0):
            mp_wrapper(self.numproc, self._ccf_inter_worker, self.chunk_numpairs)
        
        #eetime = time.time(); print('inter calculation took {:.4f} s'.format(eetime-self.start_time))
              
        self.cnt_all+=imgnum 
        self.cnt_good+=int(np.sum(isgood[:imgnum]))
        self.cnt_pairs+=self.chunk_numpairs 
        if self.verbose>=1: print('xcca: images analysed: {}, good images: {}, image pairs: {}'.format(self.cnt_all, self.cnt_good, self.cnt_pairs))
    
    # azimuthal intensity integration worker
    def _intensity_average_azimuthal_worker(self, j, numproc):
        if self.xcca_diffimages:  
            for i in range(j, self.chunk_numpairs, numproc):  
                    np.add(self.iav_average_shared[j], self.i_average_azimuthal(self.img_data_pol[i], self.mask_data_pol[i]), out=self.iav_average_shared[j])
        else:   
            for i in range(j, self.chunk_size_current, numproc):  
                if self.isgood[i]==1:
                    np.add(self.iav_average_shared[j], self.i_average_azimuthal(self.img_data_pol[i], self.mask_data_pol[i]), out=self.iav_average_shared[j])
                    
    # calculate azimuthally averaged intensity
    #
    def i_average_azimuthal(self, data_polar, mask_polar):
        sum1=data_polar.sum(axis=1) # sum of intensities
        sum2=mask_polar.sum(axis=1) # normalization
        np.divide(sum1, sum2, out=sum1, where=(sum2 != 0))
        return sum1
    
    # difference image intra-ccf worker
    def  _ccf_diff_intra_worker(self, j, numproc, numpairs):
        for i in range(j, numpairs, numproc):
            if self.xcca_qq:
                ccf_data=self.ccf_twopoint_q_q(self.img_data_pol[i])
                ccf_mask=self.ccf_twopoint_q_q(self.mask_data_pol[i])
                ccfcorrected=self.ccf_mask_correction(ccf_data, ccf_mask)
                np.add(self.diff_intra_ccf_qq_average_shared[j], ccfcorrected.real, out=self.diff_intra_ccf_qq_average_shared[j]) 
            if self.xcca_q1q2:   
                ccf_data1=self.ccf_twopoint_q1_q2(self.img_data_pol[i])
                ccf_mask1=self.ccf_twopoint_q1_q2(self.mask_data_pol[i])
                ccfcorrected1=self.ccf_mask_correction(ccf_data1, ccf_mask1)
                np.add(self.diff_intra_ccf_q1q2_average_shared[j], ccfcorrected1.real, out=self.diff_intra_ccf_q1q2_average_shared[j]) 
    
    #  inter-ccf worker
    def  _ccf_inter_worker(self, j, numproc, numpairs):
        for i in range(j, numpairs, numproc):
            if self.xcca_qq:
                ccf_data=self.ccf_twopoint_q_q(self.img_data_pol[self.imgpairs[i,0]],self.img_data_pol[self.imgpairs[i,1]])
                ccf_mask=self.ccf_twopoint_q_q(self.mask_data_pol[self.imgpairs[i,0]],self.mask_data_pol[self.imgpairs[i,1]])
                ccfcorrected=self.ccf_mask_correction(ccf_data, ccf_mask)
                np.add(self.inter_ccf_qq_average_shared[j], ccfcorrected.real, out=self.inter_ccf_qq_average_shared[j])
            if self.xcca_q1q2:   
                ccf_data1=self.ccf_twopoint_q1_q2(self.img_data_pol[self.imgpairs[i,0]],self.img_data_pol[self.imgpairs[i,1]])
                ccf_mask1=self.ccf_twopoint_q1_q2(self.mask_data_pol[self.imgpairs[i,0]],self.mask_data_pol[self.imgpairs[i,1]])
                ccfcorrected1=self.ccf_mask_correction(ccf_data1, ccf_mask1)
                np.add(self.inter_ccf_q1q2_average_shared[j], ccfcorrected1.real, out=self.inter_ccf_q1q2_average_shared[j])
                                            
    # intra-ccf worker
    def  _ccf_intra_worker(self, j, numproc, imgnum):
        for i in range(j, imgnum, numproc):  
            if self.isgood[i]==1:
                if self.xcca_qq:
                    ccf_data=self.ccf_twopoint_q_q(self.img_data_pol[i])
                    ccf_mask=self.ccf_twopoint_q_q(self.mask_data_pol[i])
                    ccfcorrected=self.ccf_mask_correction(ccf_data, ccf_mask)
                    np.add(self.intra_ccf_qq_average_shared[j], ccfcorrected.real, out=self.intra_ccf_qq_average_shared[j])
                if self.xcca_q1q2:   
                    ccf_data1=self.ccf_twopoint_q1_q2(self.img_data_pol[i])
                    ccf_mask1=self.ccf_twopoint_q1_q2(self.mask_data_pol[i])
                    ccfcorrected1=self.ccf_mask_correction(ccf_data1, ccf_mask1)
                    np.add(self.intra_ccf_q1q2_average_shared[j], ccfcorrected1.real, out=self.intra_ccf_q1q2_average_shared[j]) 
                              
    # calculate CCF and its FCs at the same resolution ring q1=q2: output n_q*n_phi array size for CCF and its FCs
    #
    def ccf_twopoint_q_q(self, data_polar1, data_polar2=np.array([])):

        # FFT of intensity for each q: Fourier coefficients of intensity
        fc_I1q = np.fft.fft(data_polar1)

        if len(data_polar2) != 0:
            fc_I2q = np.fft.fft(data_polar2) # in case inter-pattern correlation should be calculated #axis=1
        else:
            fc_I2q = fc_I1q                           # in case intra-pattern correlation should be calculated
            
        # Fourier coefficients of the two-point CCF
        fc_ccf_qq = np.multiply(fc_I1q.conjugate(), fc_I2q)
        
        # CCF as an inverse FFT of the complex conjugate product of FCs of intensity
        ccf_qq = np.fft.ifft(fc_ccf_qq)
       
        return ccf_qq
    
    
    # calculate CCF and its FCs for a (sub)set of q1 != q2 rings: output (n_q1 * n_q * n_phi) matrix size for CCF and FCs
    #
    def ccf_twopoint_q1_q2(self, data_polar1, data_polar2=np.array([])):
        
        # FFT of intensity for each q: Fourier coefficients of intensity
        fc_I1q = np.fft.fft(data_polar1)
        
        if len(data_polar2) !=0:
            fc_I2q = np.fft.fft(data_polar2) # in case inter-pattern correlation should be calculated
        else:
            fc_I2q = fc_I1q                  # in case intra-pattern correlation should be calculated
        
        #3D array, first index - q1, second - q2, third - 'frequency n'
        fc_ccf_q1q2 = np.empty((self.n_q1, self.n_q, self.n_phi), dtype=np.complex128)  
              
        for i, t in enumerate(self.q1_list):
            for j in range(self.n_q):
                fc_ccf_q1q2[i,j] = np.multiply(fc_I1q[t].conjugate(), fc_I2q[j])
           
        #3D array, first index - q1, second - q2, third - 'angle phi'
        ccf_q1q2=np.fft.ifft(fc_ccf_q1q2) #axis=2
        
        return  ccf_q1q2 
             
        
    # calculate Fourier components of the normalized CCF 
    # Output: normalized complex Fourier coefficients of the CCFs; normalization is done in such a way that DC of the CCF is ~<I0>^2
    #
    def ccf_twopoint_fc( self, ccf_data ):
        ccf_fcs=np.fft.fft(ccf_data)
        return np.true_divide(ccf_fcs, self.n_phi)
        
        
    # correct the CCF of the data by the CCF of the mask  
    #      
    def ccf_mask_correction(self, ccf_data, ccf_mask ):
        np.divide( ccf_data, ccf_mask, out=ccf_data, where=(ccf_mask !=0 ) )
        return ccf_data 

    
    # rebin 2D array 'data' by a 'binfactor' by averaging array elements; for simplicity reject remaining columns and rows of the input 'data' array which do not compose a complete bin
    #
    def image_bin(self, data): 
        shape_bin = (data.shape[0]//self.binfactor, self.binfactor, data.shape[1]//self.binfactor, self.binfactor)
        return data[:self.binfactor*shape_bin[0],:self.binfactor*shape_bin[2]].reshape(shape_bin).mean(axis=(-1,1)) # binned array
                                                   
    # spline interpolation of intensities on the polar grid    
    #
    def cartesian_to_polar(self, data_old, dpcenter, val_repl):
        
        if self.dpcentertype!=0: # if image center varies from pattern to pattern
            # arrays of x and y coordinates in Cartesian coordinate system determined from the polar grid for current dpcenter
            self.cart_x = self.cart_x_orig + dpcenter[0]
            self.cart_y = self.cart_y_orig + dpcenter[1]
        
        # array of new data points calculated using spline interpolation of the original data array;
        # val_repl should be equal to 0 for data image and mask image (in case polar coordinates sample space outside of the image boundaries)
        data_new_polar = sp.ndimage.map_coordinates( data_old, [self.cart_x.ravel(), self.cart_y.ravel()], order=self.interp_order, mode='constant', cval=val_repl, prefilter=True )
        data_new_polar.shape=(self.n_q, self.n_phi)
        
        return data_new_polar
    
    def _imgdiff_transform_to_polar_worker(self, i, numproc, img_data, mask_data, dpcenter):
        for j in range(i, self.chunk_numpairs, numproc):
        
            # mask of the difference image == combined mask
            mask=np.multiply(mask_data[self.imgpairs[j,0]],mask_data[self.imgpairs[j,1]])
                    
            # form masked difference image
            img=np.subtract(img_data[self.imgpairs[j,0]],img_data[self.imgpairs[j,1]]) 
            np.multiply(img, mask, out=img)
                    
            if self.binfactor>1: # bin the data
                img=self.image_bin(img)
                mask=self.image_bin(mask)
            
            # transform the data to polar grid    
            self.img_data_pol[j]=self.cartesian_to_polar(img, dpcenter, val_repl=0)
            self.mask_data_pol[j]=self.cartesian_to_polar(mask, dpcenter, val_repl=0)
                    
        
    def _img_transform_to_polar_worker(self, i, numproc, imgnum, img_data, mask_data, dpcenter):         
        for j in range(i, imgnum, numproc):  
            if self.isgood[j]==1:
                if self.binfactor>1: # bin the data
                    img=self.image_bin(img_data[j])
                    mask=self.image_bin(mask_data[j])
                else:
                    img=img_data[j]    
                    mask=mask_data[j]

                # transform the data to polar grid    
                self.img_data_pol[j]=self.cartesian_to_polar(img, dpcenter, val_repl=0)
                self.mask_data_pol[j]=self.cartesian_to_polar(mask, dpcenter, val_repl=0)
                
    
    # call a method with a specific algorithm for image pairs selection
    #
    def _find_image_pairs(self):
        if self.pairs_type==1:
            self.imgpairs=self._findimagepairs_static()
        elif self.pairs_type==2:
            self.imgpairs=self._findimagepairs_static_symmetric() 
        elif self.pairs_type==3:
            self.imgpairs=self._findimagepairs_static_cellresolved()
        self.chunk_numpairs=len(self.imgpairs) 
        """
        print('found {} image pairs:'.format(self.chunk_numpairs))   
        for i in range(self.chunk_numpairs):
            print('pair {}, im1={}, im2={}, celid1={}, celid2={}, isgood1={}, isgood2={}'.format(i, self.imgpairs[i,0], self.imgpairs[i,1], self.ids[self.imgpairs[i,0]], self.ids[self.imgpairs[i,1]], self.isgood[self.imgpairs[i,0]],self.isgood[self.imgpairs[i,1]]  ))
        """
                                    
    # find suitable image pairs in a usual (not pump-probe) scattering experiment for calculation of difference images or difference correlation functions
    # special condition: each image can be used only once in a pair; example pairs=[[img1, img2], [img3,img4], ..., [imgN-1, imgN]]
    #
    def _findimagepairs_static(self): 
        idxgood,=np.where(self.isgood[:self.chunk_size_current]==1) # get a list of indexes of 'good' images
        numgood=len(idxgood)
                    
        if numgood>=2:
            if numgood % 2 ==0:
                pairs=np.reshape(idxgood,(int(numgood/2), 2))
            else:
                pairs=np.reshape(idxgood[:-1],(numgood//2, 2))
                self.isgood[idxgood[-1]]=0 # label the last unused image as bad 
        else:
            if numgood==1:
                self.isgood[idxgood[0]]=0 # label the unused image as bad 
            pairs=[]
        return np.array(pairs) 
    
    
    # find suitable image pairs in a usual (not pump-probe) scattering experiment for calculation of difference images or difference correlation functions
    # special condition: every image will be used twice; example pairs=[[img1, img2], [img2,img3], ..., [imgN-1, imgN], [imgN, img1]]
    #
    def _findimagepairs_static_symmetric(self): 
        idxgood,=np.where(self.isgood[:self.chunk_size_current]==1) # get a list of indexes of 'good' images
        numgood=len(idxgood)
        pairs=[] # list of selected pairs
        if numgood>=2:
            for i in range(numgood):
                lp=idxgood[i]
                if (i+1)<numgood:
                   rp=idxgood[i+1] 
                else:
                   rp=idxgood[0]     
                pairs.append([lp,rp])  
        else:
            if numgood==1:
                self.isgood[idxgood[0]]=0 # label the image as bad       
        return np.array(pairs)
     
     
    # find suitable image pairs in a usual (not pump-probe) scattering experiment for calculation of difference images or difference correlation functions
    # special condition: only images with the same cellID can form a pair; any image can be used only once in a single pair; example pairs=[[img1pID1, img2pID1], [img3pID2,img4pID2], ...]
    # TODO: improve this function 
    def _findimagepairs_static_cellresolved(self): 
        idxgood,=np.where(self.isgood[:self.chunk_size_current]==1) # get a list of indexes of 'good' images
        numgood=len(idxgood) 
        cellids=self.ids[idxgood] # get corresponding cellIds

        isused=np.zeros(numgood)
        pairs=[] # list of selected pairs
        
        if numgood>=2:
            # find only pairs with the same cellID
            for i in range(numgood):
                lp=idxgood[i]
                try:
                    k = next(j for j in range(i+1,numgood,1) if ((cellids[j]==cellids[i]) & (isused[j]==0) & (isused[i]==0)))
                    rp=idxgood[k]
                    isused[k]=1
                except:
                     if isused[i]==0:
                        self.isgood[idxgood[i]]=0
                     #if talky>=1: print("a pair was not found for image # ", lp )
                     pass
                else:
                    pairs.append([lp,rp])
        else:
            if numgood==1:
                self.isgood[idxgood[0]]=0 # label the image as bad      
        return np.array(pairs)
        
   
    # finalize xcca - compute FCs and initiate output of xcca results
    def analysis_finalize(self):
          
      if self.cnt_good>=1:
        
        self.iav_average=self.iav_average_shared.sum(axis=0)
        if self.xcca_diffimages:
          if self.cnt_pairs>0:
            self.iav_average=np.true_divide(self.iav_average, self.cnt_pairs)
        else:
            self.iav_average=np.true_divide(self.iav_average, self.cnt_good)    
        
        if self.xcca_diffimages & (self.cnt_pairs>0):
            if self.xcca_qq:
                self.diff_intra_ccf_qq_average=self.diff_intra_ccf_qq_average_shared.sum(axis=0)
                self.diff_intra_ccf_qq_average=np.true_divide(self.diff_intra_ccf_qq_average, self.cnt_good) # we still divide by self.cnt_good=2*self.cnt_pairs, since the intra-CCF of difference images is twice larger than intra-inter CCF calculated in classical way
                self.diff_intra_ccf_fc1=self.ccf_twopoint_fc(self.diff_intra_ccf_qq_average)
            if self.xcca_q1q2:
                self.diff_intra_ccf_q1q2_average=self.diff_intra_ccf_q1q2_average_shared.sum(axis=0)
                self.diff_intra_ccf_q1q2_average=np.true_divide(self.diff_intra_ccf_q1q2_average, self.cnt_good)
                self.diff_intra_ccf_fc2=self.ccf_twopoint_fc(self.diff_intra_ccf_q1q2_average)  
                
        if self.xcca_intra:
            if self.xcca_qq:
                self.intra_ccf_qq_average=self.intra_ccf_qq_average_shared.sum(axis=0)
                self.intra_ccf_qq_average=np.true_divide(self.intra_ccf_qq_average, self.cnt_good)
                self.intra_ccf_fc1=self.ccf_twopoint_fc(self.intra_ccf_qq_average)
            if self.xcca_q1q2:
                self.intra_ccf_q1q2_average=self.intra_ccf_q1q2_average_shared.sum(axis=0)
                self.intra_ccf_q1q2_average=np.true_divide(self.intra_ccf_q1q2_average, self.cnt_good)
                self.intra_ccf_fc2=self.ccf_twopoint_fc(self.intra_ccf_q1q2_average)
                
        if self.xcca_inter & (self.cnt_pairs>0):
            if self.xcca_qq:
                self.inter_ccf_qq_average=self.inter_ccf_qq_average_shared.sum(axis=0)
                self.inter_ccf_qq_average=np.true_divide(self.inter_ccf_qq_average, self.cnt_pairs)
                self.inter_ccf_fc1=self.ccf_twopoint_fc(self.inter_ccf_qq_average)
            if self.xcca_q1q2:
                self.inter_ccf_q1q2_average=self.inter_ccf_q1q2_average_shared.sum(axis=0)
                self.inter_ccf_q1q2_average=np.true_divide(self.inter_ccf_q1q2_average, self.cnt_pairs)
                self.inter_ccf_fc2=self.ccf_twopoint_fc(self.inter_ccf_q1q2_average)
                
        if self.xcca_intra_inter & (self.cnt_pairs>0):
            if self.xcca_qq:
                self.intra_inter_ccf_qq_average=np.subtract(self.intra_ccf_qq_average, self.inter_ccf_qq_average)
                self.intra_inter_ccf_fc1=self.ccf_twopoint_fc(self.intra_inter_ccf_qq_average)
            if self.xcca_q1q2:
                self.intra_inter_ccf_q1q2_average=np.subtract(self.intra_ccf_q1q2_average, self.inter_ccf_q1q2_average)
                self.intra_inter_ccf_fc2=self.ccf_twopoint_fc(self.intra_inter_ccf_q1q2_average)                   
                    
        self.xcca_output()
            
    
    # do output of XCCA results
    #
    def xcca_output(self):
        
        if self.xcca_diffimages:
            foutiaverage=self.dirsave+'diff_iaverage.bin'
            self.write_binary_iaverage_q(foutiaverage, self.iav_average)
        else:
            foutiaverage=self.dirsave+'iaverage.bin'
            self.write_binary_iaverage_q(foutiaverage, self.iav_average)     
        
        if self.xcca_diffimages:
            if self.xcca_qq:
                if self.xcca_out_ccf:
                    foutccfqq=self.dirsave+'diff_ccf_2p_intra_qq.bin'
                    self.write_binary_ccf_twopoint_q_q(foutccfqq, self.diff_intra_ccf_qq_average) 
                if self.xcca_out_fc:
                    foutccfqq_fc=self.dirsave+'diff_ccf_2p_fc_intra_qq.bin'
                    self.write_binary_ccf_twopoint_fc_qq(foutccfqq_fc, self.diff_intra_ccf_fc1)
                if self.xcca_out_fc_ampl:    
                    foutccfqq_fc_ampl=self.dirsave+'diff_ccf_2p_fc_ampl_intra_qq.bin'
                    self.write_binary_ccf_twopoint_fc_ampl_qq(foutccfqq_fc_ampl, self.diff_intra_ccf_fc1)
            if self.xcca_q1q2:
                if self.xcca_out_ccf:
                    foutccfq1q2=self.dirsave+'diff_ccf_2p_intra'
                    self.write_binary_ccf_twopoint_q1_q2(foutccfq1q2, self.diff_intra_ccf_q1q2_average)
                if self.xcca_out_fc:
                    foutccfq1q2_fc=self.dirsave+'diff_ccf_2p_fc_intra'
                    self.write_binary_ccf_twopoint_fc_q1_q2(foutccfq1q2_fc, self.diff_intra_ccf_fc2)
                if self.xcca_out_fc_ampl:    
                    foutccfq1q2_fc_ampl=self.dirsave+'diff_ccf_2p_fc_ampl_intra'
                    self.write_binary_ccf_twopoint_fc_ampl_q1_q2(foutccfq1q2_fc_ampl, self.diff_intra_ccf_fc2)
                
        if self.xcca_intra:
            if self.xcca_qq:
                if self.xcca_out_ccf:
                    foutccfqq=self.dirsave+'ccf_2p_intra_qq.bin'
                    self.write_binary_ccf_twopoint_q_q(foutccfqq, self.intra_ccf_qq_average)
                if self.xcca_out_fc:
                    foutccfqq_fc=self.dirsave+'ccf_2p_fc_intra_qq.bin'
                    self.write_binary_ccf_twopoint_fc_qq(foutccfqq_fc, self.intra_ccf_fc1)     
                if self.xcca_out_fc_ampl:    
                    foutccfqq_fc_ampl=self.dirsave+'ccf_2p_fc_ampl_intra_qq.bin'
                    self.write_binary_ccf_twopoint_fc_ampl_qq(foutccfqq_fc_ampl, self.intra_ccf_fc1) 
            if self.xcca_q1q2:
                if self.xcca_out_ccf:
                    foutccfq1q2=self.dirsave+'ccf_2p_intra'
                    self.write_binary_ccf_twopoint_q1_q2(foutccfq1q2, self.intra_ccf_q1q2_average)
                if self.xcca_out_fc:
                    foutccfq1q2_fc=self.dirsave+'ccf_2p_fc_intra'
                    self.write_binary_ccf_twopoint_fc_q1_q2(foutccfq1q2_fc, self.intra_ccf_fc2)    
                if self.xcca_out_fc_ampl:    
                    foutccfq1q2_fc_ampl=self.dirsave+'ccf_2p_fc_ampl_intra'
                    self.write_binary_ccf_twopoint_fc_ampl_q1_q2(foutccfq1q2_fc_ampl, self.intra_ccf_fc2)
                
        if self.xcca_inter:
            if self.xcca_qq:
                if self.xcca_out_ccf:
                    foutccfqq=self.dirsave+'ccf_2p_inter_qq.bin'
                    self.write_binary_ccf_twopoint_q_q(foutccfqq, self.inter_ccf_qq_average) 
                if self.xcca_out_fc:
                    foutccfqq_fc=self.dirsave+'ccf_2p_fc_inter_qq.bin'
                    self.write_binary_ccf_twopoint_fc_qq(foutccfqq_fc, self.inter_ccf_fc1)     
                if self.xcca_out_fc_ampl:    
                    foutccfqq_fc_ampl=self.dirsave+'ccf_2p_fc_ampl_inter_qq.bin'
                    self.write_binary_ccf_twopoint_fc_ampl_qq(foutccfqq_fc_ampl, self.inter_ccf_fc1)
            if self.xcca_q1q2:
                if self.xcca_out_ccf:
                    foutccfq1q2=self.dirsave+'ccf_2p_inter'
                    self.write_binary_ccf_twopoint_q1_q2(foutccfq1q2, self.inter_ccf_q1q2_average)
                if self.xcca_out_fc:
                    foutccfq1q2_fc=self.dirsave+'ccf_2p_fc_inter'
                    self.write_binary_ccf_twopoint_fc_q1_q2(foutccfq1q2_fc, self.inter_ccf_fc2)     
                if self.xcca_out_fc_ampl:    
                    foutccfq1q2_fc_ampl=self.dirsave+'ccf_2p_fc_ampl_inter'
                    self.write_binary_ccf_twopoint_fc_ampl_q1_q2(foutccfq1q2_fc_ampl, self.inter_ccf_fc2)
                
        if self.xcca_intra_inter:
            if self.xcca_qq:
                if self.xcca_out_ccf:
                    foutccfqq=self.dirsave+'ccf_2p_intra_inter_qq.bin'
                    self.write_binary_ccf_twopoint_q_q(foutccfqq, self.intra_inter_ccf_qq_average)
                if self.xcca_out_fc:
                    foutccfqq_fc=self.dirsave+'ccf_2p_fc_intra_inter_qq.bin'
                    self.write_binary_ccf_twopoint_fc_qq(foutccfqq_fc, self.intra_inter_ccf_fc1)     
                if self.xcca_out_fc_ampl:     
                    foutccfqq_fc_ampl=self.dirsave+'ccf_2p_fc_ampl_intra_inter_qq.bin'
                    self.write_binary_ccf_twopoint_fc_ampl_qq(foutccfqq_fc_ampl, self.intra_inter_ccf_fc1)
            if self.xcca_q1q2:
                if self.xcca_out_ccf:
                    foutccfq1q2=self.dirsave+'ccf_2p_intra_inter'
                    self.write_binary_ccf_twopoint_q1_q2(foutccfq1q2, self.intra_inter_ccf_q1q2_average)
                if self.xcca_out_fc:
                    foutccfq1q2_fc=self.dirsave+'ccf_2p_fc_intra_inter'
                    self.write_binary_ccf_twopoint_fc_q1_q2(foutccfq1q2_fc, self.intra_inter_ccf_fc2)    
                if self.xcca_out_fc_ampl:    
                    foutccfq1q2_fc_ampl=self.dirsave+'ccf_2p_fc_ampl_intra_inter'
                    self.write_binary_ccf_twopoint_fc_ampl_q1_q2(foutccfq1q2_fc_ampl, self.intra_inter_ccf_fc2)                               
    
    
    # write the average intensity array of size (n_q) to a binary file; dtype='d' for double, and dtype='f' for single precision numbers
    #
    def write_binary_iaverage_q( self, fname, data, dtype='d', bo='<' ): 
        comb = np.vstack((self.qvals, data)).T
        fmt=bo+'i'+str(self.n_q*2)+dtype                    
        bin = struct.pack(fmt, self.n_q, *comb.flatten()) 
        with open( fname, "wb") as file:
            file.write( bin )
    
    
    # write the CCFs array of size (n_q * n_phi) to a binary file; dtype='d' for double, and dtype='f' for single precision numbers
    #
    def write_binary_ccf_twopoint_q_q( self, fname, data, dtype='d', bo='<' ): 
        fmt=bo+'2i'+str(self.n_q*self.n_phi)+dtype                 
        bin = struct.pack(fmt, self.n_q, self.n_phi, *data.flatten()) 
        with open( fname, "wb") as file:
            file.write( bin )
    
    
    # write the complex FCs array of size (n_q * n_max *2) to a binary file; dtype='d' for double, and dtype='f' for single precision numbers
    # the FC are written in pairs of values (real, imag)
    #
    def write_binary_ccf_twopoint_fc_qq( self, fname, data, dtype='d', bo='<' ): 
        datre=data[:,0:self.n_max].real; datim=data[:,0:self.n_max].imag
        comb = np.vstack((datre.flatten(), datim.flatten())).T
        fmt=bo+'2i'+str(self.n_q*self.n_max*2)+dtype         
        bin = struct.pack(fmt, self.n_q, self.n_max, *comb.flatten()) 
        with open( fname, "wb") as file:
            file.write( bin )
        
            
    # write the FC amplitudes array of size (n_q * n_max) to a binary file; dtype='d' for double, and dtype='f' for single precision numbers
    #
    def write_binary_ccf_twopoint_fc_ampl_qq( self, fname, data, dtype='d', bo='<' ): 
        fmt=bo+'2i'+str(self.n_q*self.n_max)+dtype  
        data1=np.abs(data)        
        bin = struct.pack(fmt, self.n_q, self.n_max, *data1[:,0:self.n_max].flatten()) 
        with open( fname, "wb") as file:
            file.write( bin )
     
            
    # write the CCFs array of size (n_q * n_phi)  for each q1 ring to a separate binary file; dtype='d' for double, and dtype='f' for single precision numbers
    #
    def write_binary_ccf_twopoint_q1_q2( self, fname_templ, data, dtype='d', bo='<' ): 
        fmt=bo+'2i'+str(self.n_q*self.n_phi)+dtype
        for i, t in enumerate(self.q1_list):
            data1=data[i]
            fname=fname_templ+'_q_'+str(t)+'_q.bin'  
            bin = struct.pack(fmt, self.n_q, self.n_phi, *data1.flatten())
            with open( fname, "wb") as file:
                file.write( bin )
                
    
    # write the complex FCs array of size (n_q * n_max *2) for each q1 ring to a separate binary file; dtype='d' for double, and dtype='f' for single precision numbers
    # the FC are written in pairs of values (real, imag)
    #
    def write_binary_ccf_twopoint_fc_q1_q2( self, fname_templ, data, dtype='d', bo='<' ): 
        fmt=bo+'2i'+str(self.n_q*self.n_max*2)+dtype         
        for i, t in enumerate(self.q1_list):
            datre=data[i, :,0:self.n_max].real; datim=data[i, :,0:self.n_max].imag
            comb = np.vstack((datre.flatten(), datim.flatten())).T
            fname=fname_templ+'_q_'+str(t)+'_q.bin'  
            bin = struct.pack(fmt, self.n_q, self.n_max, *comb.flatten()) 
            with open( fname, "wb") as file:
                file.write( bin )
    
                        
    # write the FC amplitudes array of size (n_q * n_max) for each q1 ring to a separate binary file; dtype='d' for double, and dtype='f' for single precision numbers
    #
    def write_binary_ccf_twopoint_fc_ampl_q1_q2( self, fname_templ, data, dtype='d', bo='<' ): 
        fmt=bo+'2i'+str(self.n_q*self.n_max)+dtype  
        for i, t in enumerate(self.q1_list):
            data1=np.abs(data[i])
            fname=fname_templ+'_q_'+str(t)+'_q.bin'  
            bin = struct.pack(fmt, self.n_q, self.n_max, *data1[:,0:self.n_max].flatten()) 
            with open( fname, "wb") as file:
                file.write( bin )
    
    
# multiprocessing wrapper                 
def mp_wrapper(numproc, target_proc, *proc_args): 
    jobs = []
    for i in range(numproc):
        p = mp.Process(target=target_proc, args=(i, numproc, *proc_args))   
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
