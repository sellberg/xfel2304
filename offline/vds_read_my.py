
import os
import sys
import subprocess
import time, math
import glob
import warnings
import multiprocessing as mp
import ctypes
import h5py
import numpy as np
import geom
import helper 
import xcca
import viewer as view

import matplotlib       #line 1
matplotlib.use("TkAgg") #line 2 - these two lines prevent error messages due to possible incompatibilities of Qt libraries; see here https://github.com/ContinuumIO/anaconda-issues/issues/1440
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
    
Pi = math.pi

class vds_reader():

    def __init__(self, vds_file, dir_calib, geo_file, dir_save, nproc, chunks_to_read, chunk_size, refmod, exclude_modules,
                  filterbadcells, intensity_threshold, cmode, custommask, litpixels, litpixels_threshold, averimage, avpixi, intensity_histogram, hclist, verbose):
        self.start_time = time.time()
        self.vds_file = vds_file
        self.dir_calib=dir_calib
        self.geo_file=geo_file
        self.dir_save=dir_save
        self.chunk_size = chunk_size 
        self.chunks_to_read=chunks_to_read.copy()
        self.refmod=refmod
        self.exclude_modules=exclude_modules
        self.filterbadcells=filterbadcells
        self.intensity_threshold=intensity_threshold
        self.cmode=cmode
        self.custommask=custommask
        self.litpixels=litpixels
        self.lithr=litpixels_threshold
        self.averimage=averimage
        self.avpixi=avpixi
        self.intensity_histogram=intensity_histogram
        self.hclist=hclist
        self.xcca_perform=xcca_perform
        self.verbose = verbose
        self.detshape=(16, 512, 128) # data structure for a single frame of AGIPD1M 
        
        if self.chunk_size % 32 != 0:
            print('\nWARNING: Performance is best with a multiple of 32 chunk_size')
        
        if nproc == 0:
            self.nproc = int(subprocess.check_output('nproc').decode().strip())
        else:
            self.nproc = nproc
            
        with h5py.File(vds_file, 'r') as f:
            self.dset_name = 'INSTRUMENT/'+list(f['INSTRUMENT'])[0]+'/DET/image/data'
            self.mask_name = 'INSTRUMENT/'+list(f['INSTRUMENT'])[0]+'/DET/image/mask'
            self.cellid_name = 'INSTRUMENT/'+list(f['INSTRUMENT'])[0]+'/DET/image/cellId'
            self.trainid_name = 'INSTRUMENT/'+list(f['INSTRUMENT'])[0]+'/DET/image/trainId'
            self.pulseid_name = 'INSTRUMENT/'+list(f['INSTRUMENT'])[0]+'/DET/image/pulseId'
            self.dshape = f[self.dset_name].shape
        
        print('\nResults output directory: {}'.format(self.dir_save))
        print('Input vds file: {}'.format(self.vds_file))
            
        if len(self.dshape) == 4:
            self.is_raw = False
            print('Processing calibrated data, found records for {} frames'.format(self.dshape[1]))
        elif len(self.dshape) == 5:
            self.is_raw = True
            print('Processing raw data, found records for {} frames'.format(self.dshape[1]))
        
        # chunk reading scheme related stuff
        num_chunks_max = int(math.ceil(self.dshape[1]/self.chunk_size))  # maximum possible number of data chunks
        if self.verbose>=1: print('Maximum number of data chunks ({} frames each) is {} '.format(self.chunk_size, num_chunks_max))
        self.chunks_to_read[1]=min(self.chunks_to_read[1], num_chunks_max)
        self.num_chunks=int(math.ceil((self.chunks_to_read[1]-self.chunks_to_read[0])/self.chunks_to_read[2]))
        if self.num_chunks<=0:
            print('Specified scheme ({}) for reading data in chunks cannot be used. Exiting...'.format(chunks_to_read))
            sys.exit(1)
        
        self.num_imgs_good=0 # current number of good images    
        self.num_imgs_read=0 # current number of images read from the input file 
        self.num_chk_read=0 # current number of chunks read from the input file 
           
        if self.is_raw: # open calibration files
            calib_str=self.dir_calib+'Cheetah*.h5'
            self.calib = [h5py.File(f, 'r') for f in sorted(glob.glob(calib_str))]
         
        print('AGIPD detector geometry file: {}'.format(self.geo_file))
        
        try:    
            print('Allocating shared memory arrays...')
            self.modules_shape=(self.chunk_size,)+ self.detshape # data format specificly for AGIPD1M detector
            self.modules_size=int(self.modules_shape[0]*self.modules_shape[1]*self.modules_shape[2]*self.modules_shape[3])
            self.modules_data=np.frombuffer(mp.Array(ctypes.c_double, self.modules_size).get_obj()).reshape(self.modules_shape)              # scattering data
            self.mask_data=np.frombuffer(mp.Array(ctypes.c_uint, self.modules_size).get_obj(), dtype=np.uint32).reshape(self.modules_shape)  # mask
            self.cellid_data = np.frombuffer(mp.Array(ctypes.c_uint, self.modules_shape[0]).get_obj(), dtype=np.uint32).reshape(self.modules_shape[0]) # cell IDs
            self.pulseid_data = np.frombuffer(mp.Array(ctypes.c_uint, self.modules_shape[0]).get_obj(), dtype=np.uint32).reshape(self.modules_shape[0]) # pulse IDs
            self.trainid_data = np.frombuffer(mp.Array(ctypes.c_uint, self.modules_shape[0]).get_obj(), dtype=np.uint32).reshape(self.modules_shape[0]) # train IDs
            self.is_good_data = np.frombuffer(mp.Array(ctypes.c_uint, self.modules_shape[0]).get_obj(), dtype=np.uint32).reshape(self.modules_shape[0]) # image quality identifier: 1 - good image, 0 - bad image; used to label bad images
            
            self.is_good_data_all=np.frombuffer(mp.Array(ctypes.c_uint, self.dshape[1]).get_obj(), dtype=np.uint32) # litpixels  
            
            if self.litpixels:
                self.litpix=np.frombuffer(mp.Array(ctypes.c_ulong, self.dshape[0]*self.dshape[1]).get_obj(), dtype='u8').reshape(16,-1)# litpixels  
                 
            if self.avpixi:
                self.pixintens=np.frombuffer(mp.Array(ctypes.c_double, self.dshape[0]*self.dshape[1]).get_obj()).reshape(16,-1)# average intensity/pixel
                self.pixcnts=np.frombuffer(mp.Array(ctypes.c_ulong, self.dshape[0]*self.dshape[1]).get_obj(), dtype='u8').reshape(16,-1)# counts pixels
                
            if self.averimage:
                self.img_sum=np.frombuffer(mp.Array(ctypes.c_double, self.detshape[0]*self.detshape[1]*self.detshape[2]).get_obj()).reshape(self.detshape) # powder pattern
                self.pix_sum=np.frombuffer(mp.Array(ctypes.c_uint, self.detshape[0]*self.detshape[1]*self.detshape[2]).get_obj(), dtype=np.uint32).reshape(self.detshape)  # sum of contributions  
                  
            if self.intensity_histogram[0]==True:
                self.hist_shape=(self.chunk_size, self.intensity_histogram[2])
                self.hist_vals = np.frombuffer(mp.Array(ctypes.c_double, self.hist_shape[0]*self.hist_shape[1]).get_obj()).reshape(self.hist_shape)
                
            # arrays for assembled images processing
            self.det_y, self.det_x = geom.pixel_maps_from_geometry_file(self.geo_file) # pixel coordinates of the assembled detector image
            self.img_shape_assembled=geom.apply_geom_ij_yx((self.det_x, self.det_y), np.zeros(self.detshape)).shape
            #print('\nAssembled image shape={}'.format(self.img_shape_assembled))
            self.img_shape=(self.chunk_size,)+ self.img_shape_assembled  # for assembled images
            self.img_size=int(self.img_shape[0]*self.img_shape[1]*self.img_shape[2])
            self.img_data=None # assembled image of scattering data, will be allocated later if required
            self.msk_data=None # assembled image of mask
                
            print('Finished with memory allocation')      
        except:
            print('Problems with memory allocation. Exiting...')
            sys.exit(1)
        
        # create custom detector mask 
        if self.custommask:
            self._mask_custom_agipd()
        
        if self.filterbadcells:
            self.define_bad_cells() # list of bad memroy cells
        
        # intensity histogram related stuff
        if self.intensity_histogram[0]==True:
            self.dir_hist=self.dir_save+'./histograms/'
            if os.path.exists(self.dir_hist) is not True:
                os.makedirs(self.dir_hist)
                       
    # read a chunk of data
    def read_data_chunk(self, c):
            
        self.pmin = c*self.chunk_size
        self.pmax = min (self.dshape[1], (c+1)*self.chunk_size)
        self.chunk_size_current=self.pmax-self.pmin
        self.chunk_number=c 
        self.num_chk_read+=1
        
        self.is_good_data[:self.chunk_size_current].fill(1)
        mp_wrapper(16, self._module_data_read_worker)
        self.num_imgs_read+=self.chunk_size_current # current total number of read images    
          
        if self.filterbadcells:
            vv=~np.in1d(self.cellid_data[:self.chunk_size_current], self.good_cells)
            self.is_good_data[:self.chunk_size_current]=np.where(vv,0,1)
        self.is_good_data_all[self.pmin:self.pmax]=self.is_good_data[:self.chunk_size_current]
        
        
    # read data from a file in parallel for each detector module, calibrate and perform basic corrections/masking
    def _module_data_read_worker(self, m, numproc):
        
        stime = time.time()
        if len(self.exclude_modules)>0:
            if m in self.exclude_modules:
                self.mask_data[:self.chunk_size_current,m].fill(0)    
                self.modules_data[:self.chunk_size_current,m].fill(0)   
                return
                
        self.mask_data[:self.chunk_size_current,m].fill(1)    # 1- good pixels, 0 -bad pixels
        
        with h5py.File(self.vds_file, 'r') as f:
            if self.is_raw:
                vals = f[self.dset_name][m, self.pmin:self.pmax]
                self.modules_data[:self.chunk_size_current,m]=vals[:,0]
                gain_data=vals[:,1]
            else:    
                self.modules_data[:self.chunk_size_current,m] =f[self.dset_name][m, self.pmin:self.pmax]
                msk=f[self.mask_name][m, self.pmin:self.pmax] # in the xfel data bad pixels have nonzero mask value
                self.mask_data[:self.chunk_size_current,m][msk != 0]=0
            
            cellid_data=f[self.cellid_name][self.pmin:self.pmax] # need for every module for proper calibration
            
            if m==self.refmod:
                self.trainid_data[:self.chunk_size_current]=f[self.trainid_name][self.pmin:self.pmax]
                self.pulseid_data[:self.chunk_size_current]=f[self.pulseid_name][self.pmin:self.pmax]
        
        etime = time.time()
                
        if m == self.refmod:
            self.cellid_data[:self.chunk_size_current]=cellid_data
            if self.verbose>=1:  print('\nchunk {} ({} of {}), read {} frames in {:.4f} s ({:.2f} fps)'.format(self.chunk_number, self.num_chk_read, self.num_chunks, self.chunk_size_current, etime - stime, self.chunk_size_current/(etime-stime)))  
          
        for i in range(self.chunk_size_current):
            if self.is_raw:         # calibrate raw data (offset and gain, as well as badpixel mask)
                self._calibrate_image(i, m, cellid_data[i], gain_data[i])
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                warnings.filterwarnings('ignore', r'Invalid value encountered')
          
                if self.cmode[0]=='median_asic':    # common mode correction 
                    self._cmode_median_asic(i, m)
            
            self.mask_data[i,m][np.isnan(self.modules_data[i,m])| np.isinf(self.modules_data[i,m])]=0
            self.modules_data[i,m][np.isnan(self.modules_data[i,m]) | np.isinf(self.modules_data[i,m])]=0
            
            if self.intensity_threshold is not None:    # threshold intensities
                self.mask_data[i,m][(self.modules_data[i,m]<self.intensity_threshold[0]) | (self.modules_data[i,m]>self.intensity_threshold[1])]=0        
            if self.custommask: # custom mask
                np.multiply(self.mask_data[i,m], self.maskcustom[m], out=self.mask_data[i,m])
                
            np.multiply(self.modules_data[i,m], self.mask_data[i,m], out=self.modules_data[i,m]) # apply final mask to the data 
            
    # analyse current chunk of data
    # 
    def data_chunk_analysis(self):
        
        # if necessary apply data filtering in front of this line
        self.num_imgs_good+=self.is_good_data[:self.chunk_size_current].sum(0)
        
        mp_wrapper(16, self._combined_analysis_worker)
        
        # calculate intensity histograms
        if self.intensity_histogram[0]==True: 
            self.intensity_histogram_1d()  
    
    def intensity_histogram_1d(self):
    
        if self.num_chk_read==1:
                bb,=np.where(self.is_good_data[:self.chunk_size_current]==1); jj=bb[0] # index of the first good image
                _ , bin_edges = np.histogram(self.modules_data[jj][self.mask_data[jj]==1], bins=self.intensity_histogram[2], range=self.intensity_histogram[3], weights=self.intensity_histogram[4], density=self.intensity_histogram[5])
                self.bin_centers = np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(self.intensity_histogram[2])])  
                
                str1='{}bins_1d_ch_{}_{}_range_{}_{}.bin'.format(self.dir_hist, self.chunk_number, self.intensity_histogram[6], self.intensity_histogram[3][0], self.intensity_histogram[3][1])
                helper.io.write_binary_1D_arr(str1, self.bin_centers, dtype='f') 
                
                if self.intensity_histogram[1]=='all_evol' or self.intensity_histogram[1]=='evol':
                        self.hist1D_aver=np.zeros(self.bin_centers.shape)    
                        self.hist1D_aver_shared=np.frombuffer(mp.Array(ctypes.c_double, self.nproc*self.bin_centers.shape[0]).get_obj()).reshape(self.nproc, self.bin_centers.shape[0])   
                if self.intensity_histogram[1]=='cell':
                        self.hist1D_num=np.zeros(len(self.hclist))
                        self.hist1D_aver=np.zeros((len(self.hclist),self.bin_centers.shape[0]))    
                        
        self.hist_vals.fill(0) 
            
        mp_wrapper(self.nproc, self._intensity_histogram_1d_worker)
        
        
        if self.intensity_histogram[1]=='all' or self.intensity_histogram[1]=='all_evol':
               mp_wrapper(self.nproc, self._1d_histogram_average_workerA) 

        if self.intensity_histogram[1]=='all_evol' or self.intensity_histogram[1]=='evol':
                
                mp_wrapper(self.nproc, self._1d_histogram_average_workerB)    
                self.hist1D_aver[:]=self.hist1D_aver_shared.sum(axis=0)
                   
                # plot current average histogram
                fig_hist2=plt.plot(self.bin_centers, np.divide(self.hist1D_aver, self.num_imgs_good), linewidth=2, marker="o", color='black')
                plt.yscale('log')
                plt.xlabel('ADU')
                plt.ylabel('Frequency')
                if self.intensity_histogram[6]=='image':
                    plt.savefig('{}histogram_1d_ch_{}_{}_range_{}_{}.png'.format(self.dir_hist, self.chunk_number, self.intensity_histogram[6], self.intensity_histogram[3][0], self.intensity_histogram[3][1]))
                plt.clf()
                
                str1='{}histogram_1d_ch_{}_{}_range_{}_{}.bin'.format(self.dir_hist, self.chunk_number, self.intensity_histogram[6], self.intensity_histogram[3][0], self.intensity_histogram[3][1])
                helper.io.write_binary_1D_arr(str1, np.divide(self.hist1D_aver, self.num_imgs_good), dtype='f') 
        
        if self.intensity_histogram[1]=='cell':
            for i in range(self.chunk_size_current):
                if self.is_good_data[i]==1: 
                    pos=np.where(self.hclist==self.cellid_data[i]) # location of the matching cellID
                    if len(pos[0])==1:
                        pos=pos[0][0]
                        np.add(self.hist1D_aver[pos],self.hist_vals[i],out=self.hist1D_aver[pos])
                        self.hist1D_num[pos]+=1
            """
            #test this alternative, it can be run in parallel
            for i in range(len(self.hclist)):
                idxmatch,=np.where(self.cellid_data[:self.chunk_size_current]==self.hclist[i]) # get a list of indexes of 'good' images
                if len(idxmatch)>0:
                    for k in range(len(idxmatch)):
                        np.add(self.hist1D_aver[i],self.hist_vals[idxmatch[k]],out=self.hist1D_aver[pos], where=(self.is_good_data[idxmatch[k]]==1))
                        self.hist1D_num[i]+=1  
            """
                                                    
    def _intensity_histogram_1d_worker(self, i, numproc):
        if self.intensity_histogram[6]=='image':
            for j in range(i, self.chunk_size_current, numproc):
                if self.is_good_data[j]==1:
                     self.hist_vals[j], _ = np.histogram(self.modules_data[j][self.mask_data[j]==1], bins=self.intensity_histogram[2], range=self.intensity_histogram[3], weights=self.intensity_histogram[4], density=self.intensity_histogram[5])
    
    def _1d_histogram_average_workerA(self, j, numproc):
        for i in range(j, self.chunk_size_current, numproc):
            if self.is_good_data[i]==1:
                    fig_hist1=plt.plot(self.bin_centers, self.hist_vals[i], linewidth=2, marker="o", color='red')
                    plt.yscale('log')
                    plt.xlabel('ADU')
                    plt.ylabel('Frequency')
                    if self.intensity_histogram[6]=='image':
                        plt.savefig('{}histogram_1d_trID_{}_cellID_{}_{}_range_{}_{}.png'.format(self.dir_hist, self.trainid_data[i], self.cellid_data[i], self.intensity_histogram[6], self.intensity_histogram[3][0], self.intensity_histogram[3][1]))
                    plt.clf() 
    
    def _1d_histogram_average_workerB(self, j, numproc):
        for i in range(j, self.chunk_size_current, numproc):
            np.add(self.hist1D_aver_shared[j], self.hist_vals[i], out=self.hist1D_aver_shared[j], where=(self.is_good_data[i]==1))
                                                                 
    def _combined_analysis_worker(self, m, numproc):
    
        if self.litpixels: # litpixels
            self.litpix[m, self.pmin:self.pmax] = (self.modules_data[:self.chunk_size_current,m]>=self.lithr).sum(axis=(1,2))
            
        if self.avpixi: # average intensity/pixel
            self.pixintens[m, self.pmin:self.pmax] = self.modules_data[:self.chunk_size_current,m].sum(axis=(1,2))
            self.pixcnts[m, self.pmin:self.pmax] = self.mask_data[:self.chunk_size_current,m].sum(axis=(1,2))
            
        if self.averimage: # accumulate powder pattern
            for i in range(self.chunk_size_current):
                if self.is_good_data[i]==1:
                    #np.add(self.img_sum[m], self.modules_data[i,m], where=(self.mask_data[i,m]==1), out=self.img_sum[m])
                    np.add(self.img_sum[m], self.modules_data[i,m], out=self.img_sum[m]) # assuming that masks have been applied to the data
                    np.add(self.pix_sum[m], self.mask_data[i,m], out=self.pix_sum[m])
        
    # finilize analysis: do final calculations and results output
    def analysis_finalize(self):
       
        if self.averimage: 
            img_average=np.zeros(self.detshape)
            img_average=np.divide(self.img_sum, self.pix_sum, where=(self.pix_sum>0))
            img_assembled=geom.apply_geom_ij_yx((self.det_x, self.det_y), img_average)
            print('\nAssembled image shape={}'.format(img_assembled.shape))
            out_fname = self.dir_save+os.path.basename(self.vds_file).split('_')[0] + '_powder_average_2d.bin'
            helper.io.write_binary_2D_arr(out_fname, img_assembled)
            """
            img_unassembled=self.agipd_stack(img_average)
            out_fname = self.dir_save+os.path.basename(self.vds_file).split('_')[0] + '_powder_average_2d_unassembled.bin'
            helper.io.write_binary_2D_arr(out_fname, img_unassembled)
            """
        
        if (self.intensity_histogram[0]==True) & (self.intensity_histogram[1]=='cell'):
            if self.verbose>=1:
                print('\nNumber of histograms averaged for each specified cellID:\n cellID  N\n{}\n'.format(np.vstack((self.hclist, self.hist1D_num)).T.astype(np.int32)))
            for i in range(len(self.hclist)):
              if self.hist1D_num[i]>0: 
                # plot current average hitogram
                fig_hist1=plt.plot(self.bin_centers, np.divide(self.hist1D_aver[i], self.hist1D_num[i]), linewidth=2, marker="o", color='black')
                plt.yscale('log')
                plt.xlabel('ADU')
                plt.ylabel('Frequency')
                if self.intensity_histogram[6]=='image':
                    plt.savefig('{}histogram_1d_cell_{}_{}_range_{}_{}.png'.format(self.dir_hist, self.hclist[i], self.intensity_histogram[6], self.intensity_histogram[3][0], self.intensity_histogram[3][1]))
                plt.clf()           
                
        # output results to h5 file
        out_fname = self.dir_save+os.path.basename(self.vds_file).split('.')[0] + '_results.h5'
        if os.path.exists(out_fname):
            os.remove(out_fname) 
        with h5py.File(out_fname, 'a') as f:
            if self.litpixels:
                if 'litpixels' in f: del f['litpixels']
                f['litpixels'] = self.litpix.sum(0)
            if self.avpixi:
                if 'averpixintensity' in f: del f['averpixintensity']
                iav=self.pixintens.sum(0)
                iavp=self.pixcnts.sum(0)
                np.divide(iav, iavp, where=(iavp!=0),out=iav)
                f['averpixintensity'] = iav
            if 'is_good' in f: del f['is_good'] 
            f['is_good'] = self.is_good_data_all  
            self._copy_ids(f)
                
        print('Total number of frames read from the input file: {}, effective analysis performance: {:.2f} fps \n'.format(self.num_imgs_read, self.num_imgs_read/(time.time()-self.start_time)))          
    
    # give a chunk of assembled detector images and masks 
    def make_assembled_data_chunk(self):
        if self.img_data is None:
            self.img_data=np.frombuffer(mp.Array(ctypes.c_double, self.img_size).get_obj()).reshape(self.img_shape)              
            self.msk_data=np.frombuffer(mp.Array(ctypes.c_uint, self.img_size).get_obj(), dtype=np.uint32).reshape(self.img_shape)              
        mp_wrapper(self.nproc, self._make_assembled_frames_worker)
    
    # give a chunk of assembled detector images and masks
    def _make_assembled_frames_worker(self, i, numproc):
        for j in range(i, self.chunk_size_current, numproc):
            self.img_data[j]=geom.apply_geom_ij_yx((self.det_x, self.det_y), self.modules_data[j])
            self.msk_data[j]=geom.apply_geom_ij_yx((self.det_x, self.det_y), self.mask_data[j]) # output mask for xcca -0 =bad pixel, 1 - good pixel
            np.multiply(self.img_data[j], self.msk_data[j], out=self.img_data[j])
                
    # copy frame IDs to the output file 
    def _copy_ids(self, fptr):
        f_vds = h5py.File(self.vds_file, 'r')
        dset_prefix = 'INSTRUMENT/'+list(f_vds['INSTRUMENT'])[0]+'/DET/image/'
        if 'ID/trainId' in fptr:  del fptr['ID/trainId']
        fptr['ID/trainId'] = f_vds[dset_prefix+'trainId'][:]
        if 'ID/cellId' in fptr: del fptr['ID/cellId']
        fptr['ID/cellId'] = f_vds[dset_prefix+'cellId'][:]
        if 'ID/pulseId' in fptr:  del fptr['ID/pulseId']
        fptr['ID/pulseId'] = f_vds[dset_prefix+'pulseId'][:]
        f_vds.close()
    
    # calibrate image
    def _calibrate_image(self, j, m, cellnum, gain_data):
        gain_mode = self._give_gain_mode(gain_data, m, cellnum)
        offset = np.empty(gain_mode.shape)
        gain = np.empty(gain_mode.shape)
        badpix = np.empty(gain_mode.shape)    
        for i in range(3):
            offset[gain_mode==i] = self.calib[m]['AnalogOffset'][i,cellnum][gain_mode==i]
            gain[gain_mode==i] = self.calib[m]['RelativeGain'][i,cellnum][gain_mode==i]
            badpix[gain_mode==i] = self.calib[m]['Badpixel'][i,cellnum][gain_mode==i]
        self.modules_data[j,m] = (self.modules_data[j,m] - offset)*gain
        self.mask_data[j,m][badpix != 0]=0    
                
    # give gain mode for each pixel in each cell
    def _give_gain_mode(self, gain_data, modnum, cellnum):        
        digglvl= self.calib[modnum]['DigitalGainLevel'][:,cellnum]
        high_gain = gain_data < digglvl[1]
        low_gain = gain_data > digglvl[2]
        medium_gain = (~high_gain) * (~low_gain)
        return low_gain*2 + medium_gain
    
    # common mode correction: median subtraction by 64x64 asics for each module
    def _cmode_median_asic(self, i, m): 
        data=np.multiply(self.modules_data[i,m], self.mask_data[i,m])
        data = data.reshape(8,64,2,64).transpose(1,3,0,2).reshape(64,64,16)
        data -= np.median(data, axis=(0,1))
        self.modules_data[i,m]= data.reshape(64,64,8,2).transpose(2,0,3,1).reshape(512,128)
    
    # arrange all modules to 1024 x 1024 array (stack detector modules)
    def agipd_stack(self, data):
        top = np.concatenate((data[11,:,:],data[10,:,:],data[9,:,:],data[8,:,:],data[15,:,:],data[14,:,:],data[13,:,:],data[12,:,:]),axis=-1)
        bot = np.concatenate((data[7,::-1,::-1],data[6,::-1,::-1],data[5,::-1,::-1],data[4,::-1,::-1],data[3,::-1,::-1],data[2,::-1,::-1],data[1,::-1,::-1],data[0,::-1,::-1]),axis=-1)
        return np.fliplr(np.swapaxes(np.concatenate((top,bot),axis=-2),1,0))
         
    # define manually which memroy cells should be ignored
    def define_bad_cells(self):
        npulses = 176
        good_cells = list(range(npulses))
        for r in good_cells [::-1]:
            if ((r-18) % 32 == 0) | (r>127): # get rid of every 32nd memory cells, starting at 19th cell, also remove all cells with cellID > 127
                good_cells.pop(r)
        good_cells.pop(0) # get rid of the first memory cell  
        self.good_cells=np.array(good_cells)      
        print('List of AGIPD cellIDs to be used in the analysis:')
        print(self.good_cells)
                    
    # customized AGIPD mask
    def _mask_custom_agipd(self, bwidth=1): # bwidth= border width[pixels]
        maskval=0 # it is assumed that bad pixels are labeled with zero value
        self.maskcustom=np.ones(self.detshape, dtype=np.uint8)
        self.maskcustom[5,193:193+64,64:128] = maskval #single ASIC on the 5-th module
        for i in range(16):
            self.maskcustom[i,:,:bwidth] = maskval
            self.maskcustom[i,:,-bwidth:] = maskval
            self.maskcustom[i,:,63-bwidth+1:63+bwidth+1] = maskval
            for j in range(8):
                self.maskcustom[i,j*64:j*64+bwidth,:] = maskval
                self.maskcustom[i,63+j*64-bwidth+1:63+j*64+bwidth+1,:] = maskval

                                                           
# multiprocessing wrapper                 
def mp_wrapper(numproc, target_proc, *proc_args): 
    jobs = []
    for i in range(numproc):
        p = mp.Process(target=target_proc, args=(i, numproc, *proc_args))   
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
        
        
        
        
if __name__ == '__main__':

    ############################  INPUT DATA  ############################ 
    
    vdsinp='/gpfs/exfel/data/group/theory/kurta/EuXFEL/SPB/2019_run3_p2145/test4_vds/r0018_vds_proc.h5'        # input file in vds format
    vdsinp='/gpfs/exfel/data/group/theory/kurta/EuXFEL/SPB/2019_run4_p2304/calc1_test/r0018_vds_proc.h5'        # input file in vds format
    dir_calib_constants='/gpfs/exfel/u/scratch/SPB/201802/p002145/cheetah-raw/calib/agipd/r0013-r0014-r0015/' # path to calibration constants
    file_geometry_AGIPD1M='/gpfs/exfel/data/group/theory/kurta/EuXFEL/SPB/2019_run3_p2145/kartik_a1.geom'     # geometry file for SPB
    dir_save='/gpfs/exfel/data/group/theory/kurta/EuXFEL/SPB/2019_run4_p2304/calc1_test/'                      # where to save the output data
    numproc=32  # number of processes to use in parallel computations (note, some module-defined functions always use 16 processes) 
    verbose=1   # verbose level
    
    chunksz=512               # chunk size (number of images treated as as portion)
    chunks_to_read=[0, 2, 1]  # [ch_start, ch_end, ch_step] specifies how to read chunks, starting with chunk ch_start till ch_end(excluding) with a step ch_step
    filterbadcells=True       # remove bad cells from the analysis
    refmodule=0               # reference detector module used to identify some properties of other modules; it should not be in the list 'hidemodules'
    #exclude_modules=[3,5,10]                         # list of modules which will be excluded from reading and analysis
    exclude_modules=[]
    common_mode_correction=('median_asic', None)    # ('median_asic', or None; None - optional parameter, not implemented yet); 
    intensity_threshold=(0,100) #(0,100)                     # intensity threshold, all pixels with intensities outside of the specified range will be masked
    custommask=True     # custom binary AGIPD mask created manually (see vds_reader._mask_custom_agipd()), identical for all frames
    
    litpixels=False             # determine the number of lit pixels for each image
    litpixels_threshold=25.0    # minimum intensity value for a pixel to be considered lit 
    aver_pixel_intensity=True   # determine average intensity/pixel for each image
    averimage=True              # if 2D powder pattern need to be calculated
    
    intensity_histogram=[False, 'evol', 100, (-50, 110), None, None, 'image'] # intensity_histogram=[calculate=True or False,'all', 'all_evol', 'cell', 'evol'  - regime of calculation, bins=10, range=None, weights=None, density=None, source_type='image' (the only option for the moment)
    hclist=np.arange(0, 300, 30) # list of cell numbers to be used to determine a histogram for each memory cell separately, if intensity_histogram[1]=='cell'
    
    # geometry parameters
    pixsz=200.00                      # pixel size [microns]; 
    det_sam=170.0                     # sample-detector distance [mm]
    wavelng=1.33316                      # x-ray wavelength [Angstrem] 
    #dpcenter=[574.0, 660.0]          # diffraction pattern center geom_kartik_b1
    dpcenter=[550.0, 635.0]           # center geom_kartik_a2;
    dpcentertype=0                    # dpcenter type: 0 - fixed center for all patterns (currently the only available option)
    
    #xcca parameters
    xcca_perform=False                    # True, if to perform xcca       
    xcca_diffimages=False                # True, if to use difference images to compute CCFsa; in this case only intra-CCFs for difference images are calculated; they are equivalent to intra-inter CCFs calculated in an alternative way (xcca_intra_inter)
    xcca_pairs_type=1                    # 1 - every image will be used only once, i.e. [(img1-img2), (img3-img4),(img5-img6);  2 - every image will be used twice, i.e [(img1-img2), (img2-img3),(img3-img1);
                                         # 3 - only images with the same cellID will be used to compose pairs;
    xcca_intra=True                     # True, if to calculate 'intra' CCF and its FC spectra
    xcca_inter=True                     # True, if to calculate 'inter' CCF and its FC spectra
    xcca_intra_inter=True               # True, if to calculate 'intra-inter' CCF and its FC spectra
    xcca_qq=True                         # True, if to run XCCA for q1=q2 case
    xcca_q1q2=False                       # True, if to run XCCA for q1!=q2 case
    xcca_out_ccf=True                    # True, if to prefrom output of calculated CCFs to binary files
    xcca_out_fc_ampl=False               # True, if to perform output of the amplitudes of FCs of the CCFs
    xcca_out_fc=True                     # True, if to perform output of the complex FCs of the CCFs (format: for each q and n: [re, im] values)
    
    qrange=[60, 700, 1]                        # (qstart, qstop, qstep) q-range [pixels, counted from the dp_center] to consider in xcca 
    phirange=(0.0, 2*Pi)                        # (phi_min, phi_max) angular (azimuthal) [rad] range to consider in xcca 
    n_phi=1000                                  # angular 'phi' sampling [points] of intensity
    n_max=20                                    # maximum FC order 'n' to be output in the binary file; maximum feasible is 'n_phi//2'
    q1_list=[144, 240, 320]                     # list of q1 values for which CCF at q1 != q2 should be calculated
    interp_order=0                              # spline-interpolation order (0,1, 2 or 3) to perform cartesian-to-polar transformation of diffraction patterns in xcca;
    binfactor=1                                 # if binfactor>1 images will be binned in square bins of the size [binfactor x binfactor]; the following input parameters will be divided by binfactor: dpcenter, qrange[:1], q1_list (with floor() and exclusion of repeated values); pixsz will be multiplied by binfactor
    
    # image viewer options
    view_images=True           # specifies if to view images; if this option is True, the program operates in the 'viewer' regime and does not perform any calculations (except calibration and filtering options)
    view_cmap="gnuplot2"        # colorscheme for diffraction patterns: "afmhot, "hot", "gnuplot2"
    view_clip=(0, 80)         # (min,max): (0, 800) - initial range of values to be displayed on the plots of diffraction papperns
    view_figsz=(11, 8)          # 2D figure size, e.g. (8, 8)
                                         
    ############################  CALCULATIONS  ############################ 
     
    starttime = time.time()
    
    matplotlib.rcParams["savefig.directory"] = dir_save # set the default path to save all figures
    
    if view_images:        # do not perform any analysis if viewing images
        xcca_perform=False
            
    # initialize data reader
    my_vds=vds_reader(vdsinp, dir_calib_constants, file_geometry_AGIPD1M, dir_save, numproc, chunks_to_read, chunksz, refmodule, exclude_modules, filterbadcells, intensity_threshold, common_mode_correction, custommask, litpixels, litpixels_threshold, averimage, aver_pixel_intensity, intensity_histogram, hclist, verbose) 
    
    currtime = (time.time()-starttime); print('runtime after vds initialization : {:.2f} s ({:.2f} min)'.format(currtime, currtime/60.0))
    
    # initialize xcca
    if xcca_perform: 
        my_xcca=xcca.correlation_analysis(dir_save, numproc, chunksz, qrange, phirange, n_phi, n_max, q1_list, interp_order, det_sam, wavelng, pixsz, binfactor, dpcenter, dpcentertype,
                xcca_diffimages, xcca_pairs_type, xcca_intra, xcca_inter, xcca_intra_inter, xcca_qq, xcca_q1q2, xcca_out_ccf, xcca_out_fc_ampl, xcca_out_fc, verbose)
        currtime = (time.time()-starttime); print('runtime after xcca initialization : {:.2f} s ({:.2f} min)'.format(currtime, currtime/60.0))
        
    # read and analyse data in chunks    
    for c in range(my_vds.chunks_to_read[0], my_vds.chunks_to_read[1], my_vds.chunks_to_read[2]): 
    
        my_vds.read_data_chunk(c)
        if verbose>=1: currtime = (time.time()-starttime); print('runtime after reading: {:.2f} s ({:.2f} min)'.format(currtime, currtime/60.0))
        
        if view_images:
            my_vds.make_assembled_data_chunk()
            ImageStack = view.ViewImageStack(my_vds.chunk_size_current, my_vds.img_data,  my_vds.trainid_data, my_vds.pulseid_data, my_vds.cellid_data, my_vds.is_good_data, view_cmap,  view_clip, view_figsz, dir_save)
        else:
            my_vds.data_chunk_analysis()
            if verbose>=1: currtime = (time.time()-starttime); print('runtime after vds analysis: {:.2f} s ({:.2f} min)'.format(currtime, currtime/60.0))
        
        if xcca_perform:
            my_vds.make_assembled_data_chunk()
            if verbose>=1: currtime = (time.time()-starttime); print('runtime after forming assembled images : {:.2f} s ({:.2f} min)'.format(currtime, currtime/60.0))
            my_xcca.data_chunk_analysis(my_vds.chunk_size_current, my_vds.img_data, my_vds.msk_data, my_vds.cellid_data,  my_vds.is_good_data)
            if verbose>=1: currtime = (time.time()-starttime); print('runtime after xcca : {:.2f} s ({:.2f} min)'.format(currtime, currtime/60.0))
            
    # finilize calculations, do output; in principle can be applied to any datachunk to follow the evolution of xcca results        
    my_vds.analysis_finalize()
    if xcca_perform:
        my_xcca.analysis_finalize()
    
    currtime = (time.time()-starttime); print('Program runtime: {:.2f} s ({:.2f} min)'.format(currtime, currtime/60.0))
