

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
    data_index = traits.List(traits.Int)
    data_selected = traits.List(traits.Int)

    y_offset = traits.Range(0,50)
    x_offset = traits.Range(0,50)

    reset_plot_btn = traits.Button(label='Reset plot')
    select_all_btn = traits.Button(label='All')
    select_none_btn = traits.Button(label='None')


    def __init__(self, fid):
        super(DataPlotter, self).__init__()
        data = fid.data
        self.data = data
        self.data_index = range(len(data))
        x = np.linspace(fid.params['sw_left'], fid.params['sw_left']-fid.params['sw'], len(self.data[0]))#range(len(self.data[0]))
        plot_data = ArrayPlotData(x=x, *data) #chaco class Plot require chaco class ArrayPlotData


        plot = Plot(plot_data, default_origin='bottom right')
        self.zoomtool = BetterZoom(plot, zoom_to_mouse=False, x_min_zoom_factor=1, zoom_factor=1.5)
        self.pantool = PanTool(plot)
        plot.tools.append(self.zoomtool)
        plot.tools.append(self.pantool)
        #plot.tools.append(ZoomTool(plot))
        plot.x_axis.title = 'ppm'
        plot.y_axis.visible = False
        self.renderer = plot.plot(('x', 'series1'), type='line', color='black')[0]
        self.plot = plot
        
        self.index_array = np.arange(len(self.data))
        self.y_offsets = self.index_array * self.y_offset 
        self.x_offsets = self.index_array * self.x_offset 

        self.data_selected = [0]
        self.plot.padding = [0]*4


    def reset_plot(self):
        print 'resetting plot...'
        self.x_offset, self.y_offset = 0, 0
        self.set_plot_offset(x=self.x_offset, y=self.y_offset)
        #add pan resetting

    def _reset_plot_btn_fired(self):
        self.reset_plot()

    def _select_all_btn_fired(self):
        self.data_selected = range(len(self.data))

    def _select_none_btn_fired(self):
        self.data_selected = [] 

    def set_plot_offset(self, x=None, y=None):
        if x==None and y==None:
            pass
        
        self.old_x_offsets = self.x_offsets
        self.old_y_offsets = self.y_offsets
        self.x_offsets = self.index_array * x
        self.y_offsets = self.index_array * y 
        for i,j in zip(self.index_array, self.plot.plots):
            self.plot.plots[j][0].position = [self.x_offsets[i], self.y_offsets[i]] 
        self.plot.request_redraw()

    def _y_offset_changed(self):
        self.set_plot_offset(x=self.x_offset, y=self.y_offset)
        #self.old_y_offsets = self.y_offsets
        #self.y_offsets = self.index_array * self.y_offset 
        #for i,j  in zip(self.y_offsets, self.plot.plots):
        #    self.plot.plots[j][0].position = [0,i] 
        #self.plot.request_redraw()

    def _x_offset_changed(self):
        self.set_plot_offset(x=self.x_offset, y=self.y_offset)
        #self.old_x_offsets = self.x_offsets
        #self.x_offsets = self.index_array * self.x_offset 
        #for i,j  in zip(self.x_offsets, self.plot.plots):
        #    self.plot.plots[j][0].position = [i,0] 
        #self.plot.request_redraw()

    def _data_selected_changed(self):
        print self.data_selected
        print self.plot.plots.keys()
        self.plot.delplot(*self.plot.plots)
        for i in range(len(self.data_selected)):
            self.plot.plot(('x', 'series'+str(self.data_selected[i]+1)), type='line', color='black')[0]
        self.set_plot_offset(x=self.x_offset, y=self.y_offset)

    def default_traits_view(self):
        traits_view = View(Group(Group(
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
                            Group(Group(
                                    Item('select_all_btn', show_label=False), 
                                    Item('select_none_btn', show_label=False),
                                    orientation='vertical'), 
                                  Group(
                                    Item('y_offset'),   
                                    Item('x_offset'), 
                                    orientation='vertical'), 
                                    Item('reset_plot_btn', show_label=False), 
                                    show_border=False, 
                                    orientation='horizontal')),   
                            width=1200, 
                            height=600, 
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








