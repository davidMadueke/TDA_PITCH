from matplotlib import pyplot as plt
import gudhi as gd
from gtda.plotting import plot_point_cloud


def pointCloud3d(data, block=True):
    #fig = plt.figure()
    #ax2 = fig.add_subplot(111, projection='3d')
    #ax2.scatter(data[:][0], data[:][1], data[:][2])
    #ax2.set_title("Point Cloud projected into 3 Dimensional Euclidean Space")
    #plt.show(block=block)
    return plot_point_cloud(data, dimension=None)



def getSimplex(stree, fullSTree=False, maxFiltered_values: int = 100):
    count = 0
    print('Alpha complex is of dimension ', stree.dimension(), ' - ',
          stree.num_simplices(), ' simplices - ', stree.num_vertices(), ' vertices.')

    fmt = '%s -> %.2f'
    for filtered_value in stree.get_filtration():
        if not fullSTree:
            while count < maxFiltered_values:
                print(fmt % tuple(filtered_value))
                count += 1
        else:
            print(fmt % tuple(filtered_value))

def get_betti_num(asc):
    betti_nums = asc.betti_numbers()
    print(betti_nums)
    return betti_nums

def print_diagram(diag,type=0): # type=0 diag , type = 1 barcode
    if type==0:
        gd.plot_persistence_diagram(diag)
    else:
        gd.plot_persistence_barcode(diag)
    plt.show()


def print_bigger_diagram(diag,base_size=6,added_size=40):
    #diag = [(a, (b, c)]
    """
    'b'
    blue
    'g'
    green
    'r'
    red
    """
    plt.figure(figsize=(6,6))
    colors = ['r', 'b', 'g']
    x_lst = []
    y_lst = []
    clr_lst = []
    size_lst = []
    for i in range(len(diag)):
        point = diag[i]
        x_lst.append(point[1][0])
        y_lst.append(point[1][1])
        colr = colors[point[0]]
        clr_lst.append(colr)
        size = base_size+added_size*(point[1][1]-point[1][0])
        size_lst.append(size)
    #for i in range(len(x_lst)):
     #   plt.scatter(x_lst[i],y_lst[i],s=size_lst[i],c = clr_lst[i])
    # plot xy line
    # plot y = max(death_time of blue& green) line
    y_max = max([i for i in y_lst if i != float('inf')])
    # fix infinity
    for i in range(len(y_lst)):
        if y_lst[i] == float('inf'):
            clr = clr_lst[i]
            birth = x_lst[i]
            print(x_lst[i])
            y_lst[i] = y_max*1.01  # represents infinity in a more "convinient way"
            size_lst[i] = base_size+added_size*(y_lst[i]-x_lst[i])
    y_max = max([i for i in y_lst if i != float('inf')])

    plt.scatter(x_lst,y_lst,c=clr_lst,s=size_lst)
    left,right = plt.xlim()
    down,up = plt.ylim()
    plt.xlim(left,right)
    plt.ylim(down,up)
    plt.plot([min(x_lst)-100,max(x_lst)+100],[min(x_lst)-100,max(x_lst)+100],'k-',lw=1)
    plt.hlines(y_max,min(x_lst)-100,max(x_lst)+100,color='k')

    plt.show()