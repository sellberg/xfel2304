import struct
import numpy as np

# input-output operations
#
class io:
    
    # writes a 2D array to a binary file; dtype='d' - double, 'f' - float , 'i' - integer
    #
    @classmethod
    def write_binary_2D_arr(cls, fname, data, dtype='f', bo='<' ): 
        fmt=bo+str(data.size)+dtype            
        bin = struct.pack(fmt, *data.flatten()) 
        with open( fname, "wb") as file:
            file.write( bin )
    
            
    # read 2D array of a given shape from a binary file
    #
    @classmethod
    def read_binary_2D_arr(cls, fname, shape, dtype='f', bo='<' ): 
        with open( fname, "rb") as file:
            data = file.read()
        if len(data)!=0:  
            fmt=bo+str(shape[0]*shape[1])+dtype
            dtuple=struct.unpack(fmt, data)
            data=np.asarray(dtuple).reshape(shape)
            return data
        else:
            return np.array([])      
    

    #writes a binary 1D array of N elements
    #
    @classmethod
    def write_binary_1D_arr(cls, fname, data, dtype='f', bo='<' ): 
        sz=data.shape[0]
        fmt=bo+'i'+str(sz)+dtype            
        bin = struct.pack(fmt, sz, *data.flatten()) 
        with open( fname, "wb") as file:
            file.write( bin )
    
    #read text file content line-by-line
    #
    @classmethod
    def read_line_text(cls, fname):
        with open(fname) as f:
            text_lines = f.readlines()
            text_lines = [x.strip() for x in text_lines] 
            text_lines = list(filter(None, text_lines))  
        return text_lines    
