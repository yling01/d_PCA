'''
Tim Ling

Last update: 2020.05.18
'''
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from scipy.spatial import distance

import matplotlib.font_manager as ftman
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as cb

'''
This set of functions make Ramachandran plots for the top five clusters.
The population of each cluster is printed next to the cluster Ramachandran
plot and the three letter code of is printed above each box.
The color bars is shared across all clusters.
'''
def MakeFigure(clusters, res, num_frames, NIP_ttl, NIP_clean, file_name, dir_name):
    file_name = dir_name + '/' +file_name
    TitleFP  = ftman.FontProperties(size=18)
    LegendFP = ftman.FontProperties(size=18)
    LabelFP  = ftman.FontProperties(size=20)
    num_res = len(res)
    cluster_ttl = clusters[0]
    for cluster in clusters[1:]:
        cluster_ttl = np.vstack((cluster_ttl, cluster))
    phi = cluster_ttl[:, :num_res]
    psi = cluster_ttl[:, num_res:]
    max_density = -1
    for i in range(num_res):
        phi_i = phi[:,i]
        psi_i = psi[:,i]
        max_density_temp = np.amax(calcDensity2D(phi_i, psi_i)[2])
        if max_density_temp > max_density:
            max_density = max_density_temp
    FigH = 13
    FigW = 2.5 * num_res
    NPX = num_res
    NPY = 5
    Fig = plt.figure(figsize=(FigW, FigH), dpi=300)
    left, bot, right, top = (0.20,0.0,0.90,0.70)
    HSpace = 0.3 * (top - bot) / NPY
    WSpace = 0.10 * (right - left) / NPX
    SubPlotH = (top - bot - (NPY - 1) * HSpace) / NPY
    SubPlotW = (right - left - (NPX - 1) * WSpace) / NPX
    for index, cluster in enumerate(clusters):
        num_frame_cluster = len(np.array([cluster]).flatten()) / (2 * num_res)
        population = num_frame_cluster / num_frames
        population = str(round(population * 100, 3)) + "%"
        phi = cluster[:, :num_res]
        psi = cluster[:, num_res:]
        y0 = top - index * (SubPlotH + HSpace)
        for ires in range(num_res):
            xtlv = False
            ytlv = False 
            x0 = left + ires * (SubPlotW + WSpace)
            ax = Fig.add_axes([x0, y0, SubPlotW, SubPlotH])
            if index == 4:
                ax.set_xlabel("$\phi$", fontsize=10)
                
            if ires == 0: 
                ax.set_ylabel("$\psi$", fontsize=10)
                ytlv=True
                text_l = x0 - 1.2 * SubPlotW
                text_b = y0 + SubPlotH / 2
                text_w = SubPlotW
                text_h = SubPlotH / 2
                tax = Fig.add_axes([text_l, text_b, text_w, text_h])
                tax.text(0.1, 0, population, fontsize=15)
                tax.axis('off')
            if index == 0:
                ax.set_title(res[ires], fontsize=15)
            if index == 0 and ires == 0:
                text_l = x0 - 1.2* SubPlotW
                text_b = y0 + SubPlotH
                text_w = SubPlotW
                text_h = SubPlotH / 2
                tax = Fig.add_axes([text_l, text_b, text_w, text_h])
                tax.text(0, 0, "Population", fontsize=15)
                tax.axis('off')
                
                text_l = x0 - 1.2* SubPlotW
                text_b = y0 + 2 * SubPlotH
                text_w = SubPlotW
                text_h = SubPlotH / 2
                tax = Fig.add_axes([text_l, text_b, text_w, text_h])
                NIP_print = "3D NIP All Points: " + str(round(NIP_ttl, 3)) + "\n3D NIP Clean: " + str(round(NIP_clean, 3))
                tax.text(0, 0, NIP_print, fontsize=15)
                tax.axis('off')
            phi_i = phi[:,ires]
            psi_i = psi[:,ires]

            xmidps, ymidps, dens2d = calcDensity2D(phi_i, psi_i)

            yvals, xvals = np.meshgrid(xmidps,ymidps)

            pc = MakeSubPlot(ax, xvals, yvals, dens2d, ires, xtlv, ytlv, max_density)

    cbl = left + NPX * (WSpace + SubPlotW)
    cbb = bot + SubPlotH
    cbw = 0.02
    cbh = bot + SubPlotH * 5 + HSpace * 4
    cax = Fig.add_axes([cbl, cbb, cbw, cbh])
    cb = Fig.colorbar(pc, cax=cax, orientation='vertical')
    cbticks = np.linspace(0, max_density, 5)
    cb.set_ticks(ticks=cbticks)
    cb.set_ticklabels(ticklabels=[str(round(i, 5)) for i in cbticks])
    Fig.savefig(file_name)

def SetXTicks(Axes,Ticks=None,Minor=False, FP=20, Decimals=0, Visible=False):
    if Ticks is not None:
        Axes.set_xticks(ticks=Ticks,minor=Minor)
        TLabels = [str(x) for x in np.around(Ticks,decimals=Decimals)]
        if Visible:
            Axes.set_xticklabels(labels=TLabels,minor=Minor,fontproperties=FP)
        else:
            Axes.set_xticklabels(labels=TLabels,minor=Minor,visible=Visible,fontproperties=FP)

            
def SetYTicks(Axes,Ticks=None,Minor=False, FP=20, Decimals=0, Visible=False):
    if Ticks is not None:
        Axes.set_yticks(ticks=Ticks,minor=Minor)
        TLabels = [str(x) for x in np.around(Ticks,decimals=Decimals)]
        if Visible:
            Axes.set_yticklabels(labels=TLabels,minor=Minor,fontproperties=FP)
        else:
            Axes.set_yticklabels(labels=TLabels,minor=Minor,fontproperties=FP, visible=False)

def AxesPropWrapper(Axes,XTicks=None,YTicks=None,MXTicks=None,MYTicks=None,
                    XTLDecimals=0,MXTLDecimals=0,XTLVisible=True,MXTLVisible=False,
                    YTLDecimals=0,MYTLDecimals=0,YTLVisible=True,MYTLVisible=False,
                    XYRange=[0,0,1,1], TickFP=None, MTickFP=None):

    if TickFP is None:  TickFP  = ftman.FontProperties(18)
    if MTickFP is None: MTickFP = ftman.FontProperties(10)

    SetXTicks(Axes,XTicks, Minor=False,FP=TickFP, Decimals=XTLDecimals, Visible=XTLVisible)
    SetXTicks(Axes,MXTicks,Minor=True, FP=MTickFP,Decimals=MXTLDecimals,Visible=MXTLVisible)
    SetYTicks(Axes,YTicks, Minor=False,FP=TickFP, Decimals=YTLDecimals, Visible=YTLVisible)
    SetYTicks(Axes,MYTicks,Minor=True, FP=MTickFP,Decimals=MYTLDecimals,Visible=MYTLVisible)

    left, bot, right, top = XYRange
    Axes.set_xlim(left=left,right=right)
    Axes.set_ylim(bottom=bot,top=top)

def MakeSubPlot(Axes, XVals, YVals, ColVals, ires, XTLVisible=False, YTLVisible=False, max_density=1):

    TickFP  = ftman.FontProperties(size=12)
    MTickFP = ftman.FontProperties(size=0)

    XTicks = np.array([-90,0,90]) #np.arange(-90,360, 90)
    YTicks = np.array([-90,0,90]) #np.arange(-90,360, 90)
    MXTicks = None
    MYTicks = None

    AxesPropWrapper(Axes, 
                    XTicks=XTicks, 
                    YTicks=YTicks, 
                    MXTicks=MXTicks, 
                    MYTicks=MYTicks,
                    XTLVisible=XTLVisible, 
                    YTLVisible=YTLVisible, 
                    XYRange=[-180,-180,180,180],
                    TickFP=TickFP, 
                    MTickFP=MTickFP)
    SpinceWidth=2
    [i.set_linewidth(SpinceWidth) for i in Axes.spines.values()]

    TickLineWidth=2
    for l in Axes.get_xticklines() + Axes.get_yticklines():
        l.set_markeredgewidth(TickLineWidth)

    pc = Axes.pcolormesh(XVals, YVals, ColVals, cmap=genColorMap(cmx.jet), vmax=max_density)

    return pc

def genColorMap(cmap):
    cvals = [('white')] + [(cmap(i)) for i in range(1,256)] 
    new_map = colors.LinearSegmentedColormap.from_list('new_map',cvals, N=256)
    return new_map

def calcDensity2D (Xs, Ys, Ws=None):
    assert len(Xs) == len(Ys)
    Bins = np.linspace(start=-180, stop=180, num=101)
    density2D, xedges, yedges = np.histogram2d(Xs, Ys, bins=Bins, weights=Ws, density=True)
    xmidps = 0.5 * (xedges[:-1] + xedges[1:])
    ymidps = 0.5 * (yedges[:-1] + yedges[1:])
    return xmidps, ymidps, density2D