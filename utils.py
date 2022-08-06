from matplotlib import pyplot


# -------------------------------------------------------------------------------------------------
#           Class containing Graph-specific functions
#   - Reference: (https://www.codegrepper.com/code-examples/python/tensorflow+model+history+history+plot)
# -------------------------------------------------------------------------------------------------
class Graphs ():
    def plot (self, input, title, plot_1, plot_2, x_label, y_label):
        pyplot.title(title)
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)

        pyplot.plot(input.history[plot_1])
        pyplot.plot(input.history[plot_2])

        # pyplot.figure(figsize=(10, 10))
        pyplot.legend(['Train', 'Test'], loc='upper right')
        
        pyplot.show()


    def plot_singular (self, input, title, plot, x_label, y_label):
        pyplot.title(title)
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)

        pyplot.plot(input.history[plot])
        # pyplot.figure(figsize=(10, 10))
        pyplot.show()


    def plot_diff (self, input_1, input_2, input_1_label, input_2_label, title, x_label, y_label):
        pyplot.title(title)
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)

        pyplot.plot(input_1, marker = 'X', linestyle = 'dotted')
        pyplot.plot(input_2, marker = 'o', linestyle = 'dotted')

        # pyplot.figure(figsize=(10, 10))
        pyplot.legend([input_1_label, input_2_label], loc='upper right')
        
        pyplot.show()
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

