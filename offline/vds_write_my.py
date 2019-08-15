#!/usr/bin/env python

'''Creates VDS file with synchronized AGIPD data

There is one giant VDS data set with the dimensions:
    (module, pulse_number, <A/D>, fs, ss)
where <A/D> is an extra dimension in the raw data for the analog and digital data.
Additionally it has the corresponding train, pulse and cell IDs for all the frames.
The fill value for the detector data is NaN 
(i.e. if one or more modules does not have that train)
The fill value for the cell and pulse IDs is 65536

Run `./vds.py -h` for command line options.

http://docs.h5py.org
'''

import sys,  time
import os.path as op
import glob
import argparse
import numpy as np
import h5py
import math

MAX_TRAINS_IN_FILE = 260

def truefalse(v): 
        if v.lower() in ('true'):
            return True
        elif v.lower() in ('false'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected for the progrum input argument.')
            
def create_vds(dir_exp_in, dir_save_in, runnum_in, is_raw_in, npulses, exclude_modules=[]):
    
    stime = time.time()
    
    dir_exp=dir_exp_in
    dir_save=dir_save_in
    runnum=runnum_in
    is_raw=is_raw_in
    
    parser = argparse.ArgumentParser(description='Create synchronized AGIPD VDS files')
    parser.add_argument('-i', '--in_folder', type=str , help="Input directory with the data")
    parser.add_argument('-o', '--out_folder', type=str , help="Output directory to save the data")
    parser.add_argument("-run","--runnum", type=int, help="Run number")            
    parser.add_argument("-p","--proc", type=truefalse, nargs='?', const=True, help="True, if to process calibrated data. Default true")   
                          
    args = parser.parse_args()
    
    if args.in_folder is not None: 
        dir_exp=args.in_folder
        
    if args.out_folder is not None: 
        dir_save=args.out_folder
        
    if args.proc is not None: 
        if args.proc:
            is_raw=False 
        else:
            is_raw=True     
            
    if args.runnum is not None: 
        runnum=args.runnum
        
    #npulses = 250
    folder = '{}r{:04d}/'.format(dir_exp, runnum)
    
    if is_raw==True:
        print('\nProcessing raw data for run {} from: {}'.format(runnum, folder))
    else:
        print('\nProcessing calibrated data for run {} from: {}'.format(runnum, folder))
    
         
    ntrains = -1
    ftrain = sys.maxsize
    for m in range(16):
        if len(exclude_modules)>0:
            if m in exclude_modules:
                continue
        tmin = sys.maxsize
        tmax = 0
        flist = glob.glob(folder+'/*AGIPD%.2d*.h5'%m)
        for fname in flist:
            with h5py.File(fname, 'r') as f:
                tid = f['INDEX/trainId'][:]
                if tid.max() - tid.min() > MAX_TRAINS_IN_FILE:
                    print('WARNING: Too large trainId range in %s (%d)' % (op.basename(fname), tid.max()-tid.min()))
                    continue
                tmin = min(tmin, tid.min())
                tmax = max(tmax, tid.max())
                ftrain = min(ftrain, tmin) 
        ntrains = max(ntrains, tmax-tmin) 
    ntrains = int(ntrains)+1 
    ltrain = ftrain + ntrains 
    print('found {} trains, first trainID: {}, last trainID: {}'.format( ntrains, ftrain, ltrain) )
    all_trains = np.repeat(np.arange(ftrain, ftrain+ntrains, dtype='u8'), npulses) 

    fnames = glob.glob(folder+'/*AGIPD*.h5')
    numfiles=len(fnames)
    fname=glob.glob(folder+'/*AGIPD00*.h5')[0]
    with h5py.File(fname, 'r') as f:
        det_name = list(f['INSTRUMENT'])[0]
        dshape = f['INSTRUMENT/'+det_name +'/DET/0CH0:xtdf/image/data'].shape
    print('Shape of data in', det_name, 'is', dshape[1:]) 

    if is_raw:
        out_fname = op.join(dir_save, 'r%.4d_vds_raw.h5'%runnum)
    else:
        out_fname = op.join(dir_save, 'r%.4d_vds_proc.h5'%runnum)
        
    outf = h5py.File(out_fname, 'w', libver='latest')
    outf['INSTRUMENT/'+det_name+'/DET/image/trainId'] = all_trains # create a dataset with all train numbers and write it to the h5file (considering pulse structure of the trains)

    layout_data = h5py.VirtualLayout(shape=(16, ntrains*npulses) + dshape[1:]) # create a virtual layout object with the dimensions and data type of the virtual dataset
    layout_data1 = h5py.VirtualLayout(shape=(16, ntrains*npulses) + dshape[1:]) 
    # create datasets for cellid and pulseid and fill them with a number 65535
    outdset_cid = outf.create_dataset('INSTRUMENT/'+det_name+'/DET/image/cellId',
                                      shape=(ntrains*npulses,), dtype='u2',
                                      data=65535*np.ones(ntrains*npulses, dtype='u2'))
    outdset_pid = outf.create_dataset('INSTRUMENT/'+det_name+'/DET/image/pulseId',
                                      shape=(ntrains*npulses,), dtype='u8',
                                      data=65535*np.ones(ntrains*npulses, dtype='u8'))
    cnt=0                                  
    for m in range(16):
        
        if len(exclude_modules)>0:
            if m in exclude_modules:
                continue
                
        flist = sorted(glob.glob(folder+'/*AGIPD%.2d*.h5'%m))
        for fname in flist:
            cnt+=1
            dset_prefix = 'INSTRUMENT/'+det_name+'/DET/%dCH0:xtdf/image/'%m
            with h5py.File(fname, 'r') as f:
                # Annoyingly, raw data has an extra dimension for the IDs
                #   (which is why we need the ravel)
                tid = f[dset_prefix+'trainId'][:].ravel() # equivalent to  f[dset_prefix+'trainId'][:].reshape(-1)
                # Remove the following bad data:
                #   Train ID = 0, suggesting no input from AGIPD
                #   Train ID out of range, for bit flips from the trainID server
                #   Repeated train IDs: Keep only first train with that ID
                sel = (tid>0) & (tid<ltrain) # gives true only for trIDs in the given range
                uniq, nuniq = np.unique(tid, return_counts=True, return_index=True)[1:]
                for i in uniq[nuniq>npulses]:
                    print('WARNING: Repeated train IDs in %s from ind %d' % (op.basename(fname), i))
                    sel[np.where(tid==tid[i])[0][npulses:]] = False 
                tid = tid[sel] # does actual selection of trainIDs
                indices = np.where(np.in1d(all_trains, tid))[0] 

                dset = f[dset_prefix+'data']
                
                if is_raw:
                    vsource_data = h5py.VirtualSource(dset)[sel,:,:,:]
                else:
                    vsource_data = h5py.VirtualSource(dset)[sel,:,:]
                    dset1 = f[dset_prefix+'mask']
                    vsource_data1 = h5py.VirtualSource(dset1)[sel,:,:]
                    
                layout_data[m, indices] = vsource_data
                if not is_raw:
                    layout_data1 [m, indices] = vsource_data1
                cid = f[dset_prefix+'cellId'][:].ravel()[sel]
                pid = f[dset_prefix+'pulseId'][:].ravel()[sel]
                sel_indices = np.zeros(len(all_trains), dtype=np.bool)
                sel_indices[indices] = True
                outdset_cid[sel_indices] = cid
                outdset_pid[sel_indices] = pid
                
            curtime = (time.time()-stime)   
            print('time: {:.2f} s, processed {} of {} files, {}, data shape: {} '.format(curtime, cnt, numfiles, fname, vsource_data.shape))

    outf.create_virtual_dataset('INSTRUMENT/'+det_name+'/DET/image/data', layout_data, fillvalue=np.nan)
    if not is_raw:
        outf.create_virtual_dataset('INSTRUMENT/'+det_name+'/DET/image/mask', layout_data1, fillvalue=np.nan)
    outf.close()
    print('Output vds file: {}'.format(out_fname))

if __name__ == '__main__':

    # input options will be overwritten by the command line options if any specified
    dir_exp='/gpfs/exfel/exp/SPB/201901/p002304/raw/'  # experimental input directory, e.g. '/gpfs/exfel/exp/SPB/201802/p002145/proc/' or '/gpfs/exfel/exp/SPB/201802/p002145/raw/'
    dir_save='/gpfs/exfel/data/group/theory/kurta/EuXFEL/SPB/2019_run4_p2304/vds/' # output directory 
    runnum=20    # run number
    npulses = 250 # number of pulses / train
    is_raw=True # True, if processing raw data, False otherwise 
    exclude_modules=[] # list of modules which will be excluded from reading, e.g. [], [1,3]
    
    startTime = time.time()
    create_vds(dir_exp, dir_save, runnum, is_raw, npulses, exclude_modules)
    TotalTime = (time.time()-startTime)
    print('Program runtime: {} s'.format(str(math.ceil(TotalTime))))
