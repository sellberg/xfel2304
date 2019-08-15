import sys
import numpy as np
import matplotlib       #line 1
matplotlib.use("TkAgg") #line 2 - these two lines prevent error messages due to possible incompatibilities of Qt libraries; see here https://github.com/ContinuumIO/anaconda-issues/issues/1440
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox


#Image viewer
#
class ViewImageStack(object): # interactive image viewer for a stack of images

    def __init__(self, chunk_size_current, img_data, trainid_data, pulseid_data, cellid_data, is_good_data, view_cmap,  view_clip, view_figsz, dir_save):

        self.fg = plt.figure(figsize=view_figsz)
        self.cmap= view_cmap
        self.clmin = view_clip[0]
        self.clmax = view_clip[1]
        self.dirsave=dir_save

        self.imtot=chunk_size_current  # total number of images in the file
        self.X=img_data
        self.trainid_data = trainid_data
        self.pulseid_data = pulseid_data
        self.cellid_data = cellid_data
        self.imtype = is_good_data
        self.ind=0 #initial image to show, e.g, 0 or self.imtot//2

        self.mindef=self.clmin # default values used for error handling
        self.maxdef=self.clmax
        clrinact='lightgray'  # 'papayawhip' 'lightgray'
        clrhover='darkgray'      # 'salmon' 'darkgray'

        # axes for all visual components
        self.ax = self.fg.add_subplot(111)
        self.cbax = self.fg.add_axes([0.85, 0.1, 0.03, 0.77])     # for colorbar [left, bottom, width, height]
        self.resax = self.fg.add_axes([0.05, 0.83, 0.1, 0.05])    # for 'Reset' button
        self.rax = self.fg.add_axes([0.05, 0.72, 0.1, 0.1], facecolor=clrinact)      # for radiobuttons
        self.minax = self.fg.add_axes([0.04, 0.65, 0.12, 0.05])  # min and max values for datarange
        self.maxax = self.fg.add_axes([0.04, 0.59, 0.12, 0.05])
        self.imnax = self.fg.add_axes([0.07, 0.23, 0.1, 0.04], facecolor=clrinact)      # chose image number
        self.savebut = self.fg.add_axes([0.05, 0.30, 0.1, 0.05])    # for 'Save image' button
        self.exitax = self.fg.add_axes([0.05, 0.93, 0.1, 0.05])    # for 'Exit' button

        # visual components
        self.im = self.ax.imshow(self.X[self.ind, : , :], cmap=self.cmap, clim=(self.clmin,self.clmax))
        self.cls=plt.colorbar(self.im, cax=self.cbax)
        self.averbtn = Button(self.savebut, 'Save png', color=clrinact)
        self.resbt = Button(self.resax, 'Reset view', color=clrinact)
        self.exitbt = Button(self.exitax, 'Exit', color=clrinact)
        self.tbmin = TextBox(self.minax, 'min:', initial=str(self.clmin), color=clrinact)
        self.tbmax = TextBox(self.maxax, 'max:', initial=str(self.clmax), color=clrinact)
        self.rb= RadioButtons(self.rax, ('lin', 'log'), active=0, activecolor='red')
        self.imnum = TextBox(self.imnax, 'jump to \n image #', initial=str(self.ind+1), color=clrinact)

        strID="Image information:\nTrainID: "+str(self.trainid_data[self.ind])+"\nPulseID: "+str(self.pulseid_data[self.ind])+"\nCellID: "+str(self.cellid_data[self.ind])+"\nImage type: "+str(self.imtype[self.ind])
        self.textID=self.fg.text(0.03, 0.085, strID)

        self.fg.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fg.canvas.mpl_connect('button_press_event', self.on_press)
        self.fg.canvas.mpl_connect('key_press_event', self.on_keypress)

        self.update()
        plt.show(block=True)


    def on_press(self, event):

        if event.inaxes == self.rax.axes:
            self.update()

        if event.inaxes == self.resax.axes: # 'reset view' button  : default datarange, linear scale, first image
            self.clmin=self.mindef
            self.clmax=self.maxdef
            self.tbmin.set_val(self.clmin)
            self.tbmax.set_val(self.clmax)
            self.rb.set_active(0)
            self.ind = 0
            print('updated data range is [%s , %s ]' %(self.clmin, self.clmax))
            self.update()

        if event.inaxes ==self.exitax:
            sys.exit(0) # exit program

        if event.inaxes == self.savebut.axes: # 'save image' button

            items=[self.ax, self.cbax, self.cbax.get_yaxis().get_label(), self.ax.get_xaxis().get_label(), self.ax.get_yaxis().get_label()]
            bbox = Bbox.union([item.get_window_extent() for item in items])
            extent = bbox.transformed(self.fg.dpi_scale_trans.inverted())
            strim='img_trID_{:d}_plID_{:d}_ceID_{:d}_type_{:d}.png'.format(self.trainid_data[self.ind], self.pulseid_data[self.ind], self.cellid_data[self.ind], self.imtype[self.ind])
            self.fg.savefig(self.dirsave+strim, bbox_inches=extent)


    def on_keypress(self, event):

        #if (event.inaxes == self.minax.axes) | (event.inaxes == self.maxax.axes):
        if event.key == 'enter': #one may face bad script behaviour here if some specific keyboard button like 'tab' has been pressed
           try:
                self.clmin=float(self.tbmin.text)
                self.clmax=float(self.tbmax.text)
                #print('updated data range is [%s , %s ]' %(self.clmin, self.clmax))
           except:
                print('inappropriate values for data range specified')
                self.tbmin.set_val(self.clmin)
                self.tbmax.set_val(self.clmax)
           self.update()

        if (event.inaxes == self.imnax):
         if event.key == 'enter': #one may face bad script behaviour here if some specific keyboard button like 'tab' has been pressed
           try:
                val=int(self.imnum.text)
                if (val>=1) & (val<=self.imtot):
                    self.ind=val-1
                else:
                    print('inappropriate image number specified')
                    self.imnum.set_val(str(self.ind+1))
           except:
                print('inappropriate image number specified')
                self.imnum.set_val(str(self.ind+1))
           self.update()


    def on_scroll(self, event):

        if event.button == 'up':
            self.ind = (self.ind + 1) % self.imtot
        else:
            self.ind = (self.ind - 1) % self.imtot
        self.update()


    def update(self):

        self.ax.set_title('image # %s out of %s \n (use scroll wheel to navigate through images)' % (self.ind+1, self.imtot))

        # choose linear or log scale
        if self.rb.value_selected=='lin':
            self.im.set_data(self.X[self.ind, :, : ])
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.im.set_data(np.log10(self.X[self.ind, :, : ]))
                #self.im.set_clim(np.log10(self.clmin), np.log10(self.clmax))  #adjust scaling
        # update display representation
        self.im.set_clim(self.clmin, self.clmax)

        strID="Image information:\nTrainID: "+str(self.trainid_data[self.ind])+"\nPulseID: "+str(self.pulseid_data[self.ind])+"\nCellID: "+str(self.cellid_data[self.ind])+"\nImage type: "+str(self.imtype[self.ind])
        self.textID.set_text(strID)

        self.cls.draw_all()
        self.im.axes.figure.canvas.draw()

    def __del__(self):

        self.fg.canvas.mpl_disconnect(self.on_scroll)
        self.fg.canvas.mpl_disconnect(self.on_press)
        self.fg.canvas.mpl_disconnect(self.on_keypress)



if __name__ == '__main__':

    # image viewer options
    view_images=True           # specifies if to view images; if this option is True, the program operates in the 'viewer' regime and does not perform any calculations (except calibration and filtering options)
    view_cmap="gnuplot2"        # colorscheme for diffraction patterns: "afmhot, "hot", "gnuplot2"
    view_clip=(0, 1)         # (min,max): (0, 800) - initial range of values to be displayed on the plots of diffraction papperns
    view_figsz=(11, 8)          # 2D figure size, e.g. (8, 8)
    
    numimgs=10; # number of images
    szx=1000; szy=800; # image dimensions
    dir_save='./' # directory to save images
    
    images=np.random.rand(numimgs, szx, szy) # stack of images
    trainid_data=np.ones(numimgs, dtype=np.int) # trainIDs
    pulseid_data=np.arange(numimgs, dtype=np.int) #pulseIDs
    cellid_data=np.arange(numimgs, dtype=np.int) #cellIDs
    is_good_data=np.zeros(numimgs, dtype=np.int) #1- good, 0 -bad image
    
    matplotlib.rcParams["savefig.directory"] = dir_save # set the default path to save all figures
    
    ImageStack = ViewImageStack(numimgs, images, trainid_data, pulseid_data, cellid_data, is_good_data, view_cmap,  view_clip, view_figsz, dir_save)
        
