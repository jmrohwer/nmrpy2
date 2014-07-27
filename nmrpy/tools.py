

#from traits.api import HasTraits, Instance, Enum, Range, List, Int, Array
import traits.api as traits
from traitsui.api import View, Item, CheckListEditor, TabularEditor, HGroup, UItem, TabularEditor
from chaco.api import Plot, MultiArrayDataSource, ArrayPlotData
from chaco.tools.api import PanTool, ZoomTool
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


    def __init__(self, fid):
        super(DataPlotter, self).__init__()
        data = fid.data
        self.data = data
        self.data_index = range(len(data))
        x = range(len(self.data[0]))
        plot_data = ArrayPlotData(x=x, *data) #chaco class Plot require chaco class ArrayPlotData

        plot = Plot(plot_data)
        plot.tools.append(PanTool(plot))
        plot.tools.append(ZoomTool(plot))
        self.renderer = plot.plot(('x', 'series1'), type='line', color='black')[0]

        self.plot = plot
        self.data_selected = [0]


    def _data_selected_changed(self):
        self.plot.delplot(*self.plot.plots)
        self.plot.request_redraw()
        for i in range(len(self.data_selected)):
             self.plot.plot(('x', 'series'+str(self.data_selected[i]+1)), type='line', color='black')[0]

    def default_traits_view(self):
        traits_view = View(HGroup(
                            Item('data_index',
                                  editor     = TabularEditor(
                                                   show_titles  = False,
                                                   selected     = 'data_selected',
                                                   editable     = False,
                                                   multi_select = True,
                                                   adapter      = MultiSelectAdapter()),
                            width=0.05, show_label=False, has_focus=True),
                            Item('plot', editor=ComponentEditor(), show_label=False)),
                            width=1200, height=400, resizable=True, title='blah')
        return traits_view



class MainWindow(traits.HasTraits):
    data_plotter = traits.Instance(DataPlotter,())

    traits_view = View(
                        Item('data_plotter', show_label=False),
                        width=200, height=400, resizable=True, title='blah')
    def __init__(self, fid):
        super(MainWindow,self).__init__()
        self.data_plotter = DataPlotter(fid)


if __name__ == "__main__":
    print 'This module has to be imported as a submodule of nmrpy'
    pass








