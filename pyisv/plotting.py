import proplot as pplt
from matplotlib.patches import Circle
import pandas as pd
import xarray as xr
import numpy as np


def process_pcs_phase_plot(datestrt, datelast, pc):  
    """
    Extract and process principal components (PCs) data for a specified date range.
    This function takes as input the starting and ending dates and the principal components 
    data. It processes the data to extract the PCs, amplitude, and corresponding dates, months,
    and days for the given date range.
    
    Parameters:
    datestrt (int): The starting date in the format of YYYYMMDD.
    
    datelast (int): The ending date in the format of YYYYMMDD.
    
    pc (DataFrame): The principal components data with a datetime index.
    
    Returns:
    tuple: A tuple containing the following data for the specified date range:
    
    PC1 (array): Principal component 1 values.
    
    PC2 (array): Principal component 2 values.
    
    dates (array): Dates in the format of YYYYMMDD.
    
    months (array): Months as integers (1-12).
    
    days (array): Days as integers (1-31).
    """
    
    # Extract year, month, day, and hour from the datetime index
    yyyy, mm, dd, hh = pc.time.dt.year, pc.time.dt.month, pc.time.dt.day, pc.time.dt.hour
    # Extract the first two principal components
    pc1, pc2 = pc[:, 0].values, pc[:, 1].values
    # Calculate the amplitude
    amp = np.sqrt(pc1**2 + pc2**2)
    # Create a dictionary with the extracted data
    data = {'yyyy': yyyy, 'mm': mm, 'dd': dd, 'hh': hh, 'pc1': pc1, 'pc2': pc2, 'amp': amp}
    # Create a DataFrame using the dictionary
    data = pd.DataFrame(data)
    # Calculate the dates, months, and days as integers
    DATES = data.yyyy.values * 10000 + data.mm.values * 100 + data.dd.values
    MONTHS = data.mm.values
    DAYS = data.dd.values
    # Find the indices for the specified date range
    istrt = np.where(DATES == datestrt)[0][0]
    ilast = np.where(DATES == datelast)[0][0]
    # Subset data to only the dates we want to plot
    dates = DATES[istrt:ilast + 1]
    months = MONTHS[istrt:ilast + 1]
    days = DAYS[istrt:ilast + 1]
    PC1 = data.pc1.values[istrt:ilast + 1]
    PC2 = data.pc2.values[istrt:ilast + 1]

    return PC1, PC2, dates, months, days


def phase_diagram(indexname, datestrt, datelast, pc, plotname, plottype):
    """
    Credits: The function is adapted from METcalcpy
    
    Creates a phase diagram for the given index using Principal Components (PC1 and PC2) and saves it to a file.
    
    Parameters:
    indexname (str): Name of the index for which the phase diagram is to be plotted.
    
    datestrt (int): The starting date in the format of YYYYMMDD.
    
    datelast (int): The ending date in the format of YYYYMMDD.
    
    pc (DataFrame): The principal components data with a datetime index.
    
    plotname (str): Name of the output file to save the plot.
    
    plottype (str): File format for the output plot (e.g., 'png', 'pdf', 'jpeg').
    
    Returns:
    None
    """
    
    PC1, PC2, dates, months, days = process_pcs_phase_plot(datestrt, datelast, pc)
    
    # Reverse PC1 and PC2 if indexname is 'OMI'
    if indexname == 'OMI':
        PC1, PC2 = PC2, -PC1.copy()

    monthnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = ['black', 'gold', 'tab:purple', 'tab:red', 'darkgreen', 'tab:orange', 'tab:blue', 'tab:grey',
              'tab:green', 'tab:pink', 'tab:olive', 'tab:cyan']

    nMon = 0
    monName = [monthnames[months[0]-1]]
    monCol = [colors[months[0]-1]]
    
    pplt.rc.update({'meta.width':1, 'ticklabelweight':'bold', 'axeslabelweight':'bold','titleweight':'bold','fontname':'Noto Sans','fontsize':12,'titlesize':14})
    # Create a plot with reference width and height
    fig, ax = pplt.subplots(refwidth=4, refheight=4)

    # Draw the reference lines and circle
    ax.plot([-4, 4], [-4, 4], [-4, 4], [4, -4], [-4, 4], [0, 0], [0, 0], [-4, 4],
            lw=0.2, ls='--', color='k')
    ax.format(xlim=(-4.0, 4.0), ylim=(-4.0, 4.0), aspect='equal')
    ax.add_patch(Circle((0, 0), radius=1.0, fc='k', ec='k', alpha=0.2))

    labelday = 5
    ax.scatter(PC1[0], PC2[0], color='k', marker='o', s=25)

    # Draw the lines connecting the data points and add day labels
    for im, day in enumerate(days[:-1]):
        lcolor = colors[months[im] - 1]
        ax.plot(PC1[im:im + 2], PC2[im:im + 2], '-', color=lcolor, alpha=1.0)
        ax.scatter(PC1[im], PC2[im], color='k', alpha=1.0, marker='o', s=4)

        if day % labelday == 0:
            ax.text(PC1[im], PC2[im], str(day), color='k')

        # Update month labels
        if im > 0 and days[im] == 1:
            nMon = nMon + 1
            monName.append(monthnames[months[im] - 1])
            monCol.append(colors[months[im]-1])  

    ax.scatter(PC1[-1], PC2[-1], color='k', alpha=1.0, marker='o', s=4)
    ax.format(title=f"{indexname} {dates.min()} to {dates.max()}")

    # Add phase labels to the plot
    ax.text(0.55, 0.99, "Phase 7 (Western Pacific) Phase 6", ha='center', va='top', transform=ax.transAxes)
    ax.text(0.55, 0.01, "Phase 2 (Indian) Phase 3", ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.99, 0.5, "Phase 5 (Maritime) Phase 4", ha='right', va='center', rotation=-90, transform=ax.transAxes)
    ax.text(0.01, 0.5, "Phase 1 (Western Hem, Africa) Phase 8", ha='left', va='center', rotation=90, transform=ax.transAxes)

    # Add month labels at the bottom of the plot
    xstrt = 0.01
    for im in range(nMon + 1):
        ax.text(xstrt, 0.01, monName[im], color=monCol[im], ha='left', va='bottom', transform=ax.transAxes)
        xstrt += 0.1

    # Save the figure to a file
    fig.savefig(plotname+'.'+plottype, format=plottype, dpi=150)
         
