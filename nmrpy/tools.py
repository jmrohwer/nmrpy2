

#from traits.api import HasTraits, Instance, Enum, Range, List, Int, Array
import traits.api as traits
from traitsui.api import View, Item, CheckListEditor, TabularEditor, HGroup, UItem, TabularEditor, Group
from chaco.api import Plot, MultiArrayDataSource, ArrayPlotData
from chaco.tools.api import PanTool, ZoomTool, BetterZoom
from enable.component_editor import ComponentEditor
import numpy as np
from traitsui.tabular_adapter import TabularAdapter
from matplotlib import cm

class MultiSelectAdapter(TabularAdapter):
    """ This adapter is used by both the left and right tables
    """

    # Titles and column names for each column of a table.
    # In this example, each table has only one column.
    columns = [ ('index', 'myvalue') ]

    # Magically named trait which gives the display text of the column named
    # 'myvalue'. This is done using a Traits Property and its getter:
    myvalue_text = traits.Property

    # The getter for Property 'myvalue_text' simply takes the value of the
    # corresponding item in the list being displayed in this table.
    # A more complicated example could format the item before displaying it.
    def _get_myvalue_text(self):
        return self.item

class DataPlotter(traits.HasTraits):
    plot = traits.Instance(Plot) #the attribute 'plot' of class DataPlotter is a trait that has to be an instance of the chaco class Plot.
    plot_data = traits.Instance(ArrayPlotData)
    data_index = traits.List(traits.Int)
    data_selected = traits.List(traits.Int)

    y_offset = traits.Range(0,50, value=0)
    x_offset = traits.Range(-15,15, value=0)
    y_scale = traits.Range(1e-3,2.0, value=1.0)

    reset_plot_btn = traits.Button(label='Reset plot')
    select_all_btn = traits.Button(label='All')
    select_none_btn = traits.Button(label='None')

    #processing
    lb = traits.Float(10.0)
    lb_btn = traits.Button(label='Apodisation')
    lb_plt_btn = traits.Button(label='Plot Apod.')
    zf_btn = traits.Button(label='Zero-fill')
    ft_btn = traits.Button(label='Fourier transform')

    ph_auto_btn = traits.Button(label='Auto phase:\nall')
    ph_auto_single_btn = traits.Button(label='Auto phase:\nselected')
    ph_man_btn = traits.Button(label='Manual phase')

    def __init__(self, fid):
        super(DataPlotter, self).__init__()
        self.fid = fid
        data = fid.data
        self.data_index = range(len(data))
        self.x = np.linspace(fid.params['sw_left'], fid.params['sw_left']-fid.params['sw'], len(self.fid.data[0]))#range(len(self.data[0]))
        self.plot_data = ArrayPlotData(x=self.x, *np.real(data)) #chaco class Plot require chaco class ArrayPlotData


        plot = Plot(self.plot_data, default_origin='bottom right')
        self.zoomtool = BetterZoom(plot, zoom_to_mouse=False, x_min_zoom_factor=1, zoom_factor=1.5)
        self.pantool = PanTool(plot)
        plot.tools.append(self.zoomtool)
        plot.tools.append(self.pantool)
        plot.x_axis.title = 'ppm'
        plot.y_axis.visible = False
        self.renderer = plot.plot(('x', 'series1'), type='line', line_width=0.5, color='black')[0]
        #plot.x_axis.tick_label_position = 'inside'
        self.old_y_scale = self.y_scale
        self.plot = plot
        self.index_array = np.arange(len(self.fid.data))
        self.y_offsets = self.index_array * self.y_offset 
        self.x_offsets = self.index_array * self.x_offset 
        self.data_selected = [0]
        self.plot.padding = [0,0,0,35]

    def _y_scale_changed(self):
        self.set_y_scale(scale=self.y_scale)

    def set_y_scale(self, scale=y_scale):
        self.plot.value_range.high /= scale/self.old_y_scale
        self.plot.request_redraw()
        self.old_y_scale = scale

    def reset_plot(self):
        self.x_offset, self.y_offset = 0, 0
        self.y_scale = 1.0
        self.plot.index_range.low, self.plot.index_range.high = [self.x[-1], self.x[0]]
        self.plot.value_range.low = self.plot.data.arrays['series%i'%(self.data_selected[0]+1)].min()
        self.plot.value_range.high = self.plot.data.arrays['series%i'%(self.data_selected[0]+1)].max()
        #add pan resetting

    def _reset_plot_btn_fired(self):
        print 'resetting plot...'
        self.reset_plot()

    def _select_all_btn_fired(self):
        self.data_selected = range(len(self.fid.data))

    def _select_none_btn_fired(self):
        self.data_selected = [] 

    def set_plot_offset(self, x=None, y=None):
        if x==None and y==None:
            pass
        
        self.old_x_offsets = self.x_offsets
        self.old_y_offsets = self.y_offsets
        self.x_offsets = self.index_array * x
        self.y_offsets = self.index_array * y 
        for i in np.arange(len(self.plot.plots)):
            self.plot.plots['plot%i'%i][0].position = [self.x_offsets[i], self.y_offsets[i]] 
        self.plot.request_redraw()

    def _y_offset_changed(self):
        self.set_plot_offset(x=self.x_offset, y=self.y_offset)

    def _x_offset_changed(self):
        self.set_plot_offset(x=self.x_offset, y=self.y_offset)

    #for some mysterious reason, selecting new data to plot doesn't retain the plot offsets even if you set them explicitly
    def _data_selected_changed(self):
        self.plot.delplot(*self.plot.plots)
        self.plot.request_redraw()
        for i in self.data_selected:
            self.plot.plot(('x', 'series%i'%(i+1)), type='line', line_width=0.5, color='black')
        self.reset_plot()
        
    #processing buttons
    
    #plot the current apodisation function based on lb, and do apodisation
    #=================================================
    def _lb_plt_btn_fired(self):
        if self.fid._ft:
            return
        if 'lb1' in self.plot.plots:
        #if 'lb1' in self.plot_data.arrays:
            #self.plot_data.del_data('lb1')
            self.plot.delplot('lb1')
            self.plot.request_redraw()
            return 
        self.plot_lb()

    def plot_lb(self):
        if self.fid._ft:
            return
        lb_data = self.fid.data[self.data_selected[0]] 
        lb_plt = np.exp(-np.pi*np.arange(len(lb_data))*(self.lb/self.fid.params['sw_hz'])) * lb_data[0]
        self.plot_data.set_data('lb1', np.real(lb_plt))
        self.plot.plot(('x', 'lb1'), type='line', name='lb1', line_width=1, color='blue')[0]
        self.plot.request_redraw()
       
    def _lb_changed(self):
        if self.fid._ft:
            return
        lb_data = self.fid.data[self.data_selected[0]] 
        lb_plt = np.exp(-np.pi*np.arange(len(lb_data))*(self.lb/self.fid.params['sw_hz'])) * lb_data[0]
        self.plot_data.set_data('lb1', np.real(lb_plt))
         
    def _lb_btn_fired(self):
        if self.fid._ft:
            return
        self.fid.emhz(self.lb)
        self.update_plot_data_from_fid()
    #=================================================

    def _zf_btn_fired(self):
        if self.fid._ft:
            return
        if 'lb1' in self.plot.plots:
            self.plot.delplot('lb1')
        self.fid.zf()
        self.update_plot_data_from_fid()

    def _ft_btn_fired(self):
        if 'lb1' in self.plot.plots:
            self.plot.delplot('lb1')
        self.fid.ft()
        self.update_plot_data_from_fid()
        self.reset_plot()

    def _ph_auto_btn_fired(self):
        if self.fid._ft:
            self.fid.phase_auto(discard_imaginary=False)
        self.update_plot_data_from_fid()

    def _ph_auto_single_btn_fired(self):
        if self.fid._ft:
            for i in self.data_selected:
                self.fid._phase_area_single(i)
        self.update_plot_data_from_fid()

    def update_plot_data_from_fid(self):
        self.x = np.linspace(self.fid.params['sw_left'], self.fid.params['sw_left']-self.fid.params['sw'], len(self.fid.data[0]))
        self.plot_data.set_data('x', self.x)
        for i in self.index_array:
            self.plot_data.set_data("series%i"%(i+1), np.real(self.fid.data[i]))
        self.plot.request_redraw()


    def default_traits_view(self):
        traits_view = View(Group(
                            Group(
                                Item('data_index',
                                      editor     = TabularEditor(
                                                       show_titles  = False,
                                                       selected     = 'data_selected',
                                                       editable     = False,
                                                       multi_select = True,
                                                       adapter      = MultiSelectAdapter()),
                                width=0.02, show_label=False, has_focus=True),
                                Item(   'plot', 
                                        editor=ComponentEditor(), 
                                        show_label=False), 
                                        padding=0,  
                                        show_border=False, 
                                        orientation='horizontal'),
                            Group(Group(Group(
                                    Item('select_all_btn', show_label=False), 
                                    Item('select_none_btn', show_label=False),
                                    Item('reset_plot_btn', show_label=False), 
                                    orientation='vertical'), 
                                  Group(
                                    Item('y_offset'),   
                                    Item('x_offset'), 
                                    Item('y_scale', show_label=True),
                                    orientation='vertical'), orientation='horizontal', show_border=True, label='Plotting'), 
                                  Group(
                                    Group(
                                    Item('lb', show_label=False, format_str='%.2f Hz'), 
                                    Item('lb_btn', show_label=False),
                                    Item('lb_plt_btn', show_label=False),
                                    orientation='horizontal'),
                                    Group(
                                    Item('zf_btn', show_label=False), 
                                    Item('ft_btn', show_label=False),
                                    orientation='horizontal'),
                                  Group(
                                    Item('ph_auto_btn', show_label=False),
                                    Item('ph_auto_single_btn', show_label=False),
                                    Item('ph_man_btn', show_label=False),
                                    orientation='horizontal'),
                                    show_border=True, label='Processing' 
                                    ), 

                                    show_border=True, 
                                    orientation='horizontal')
                    
                                    ),   
                            width=1.0, 
                            height=0.6, 
                            resizable=True, 
                            title='NMRPy')
        return traits_view



#class MainWindow(traits.HasTraits):
#    data_plotter = traits.Instance(DataPlotter,())
#
#    traits_view = View(
#                        Item('data_plotter', show_label=False),
#                        width=200, height=400, resizable=True, title='blah')
#    def __init__(self, fid):
#        super(MainWindow,self).__init__()
#        self.data_plotter = DataPlotter(fid)


if __name__ == "__main__":
    print 'This module has to be imported as a submodule of nmrpy'
    pass








